# =========================
# train.py  (space-safe checkpoints)
# =========================

import os
# Silence TensorFlow/XLA noise if it gets imported indirectly by some deps
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tqdm import tqdm
import mlflow

import data
from model import BertMultiLabelClassifier, freeze_base_layers, unfreeze_all_layers


# -----------------------------
# Environment & Drive utilities
# -----------------------------
def in_colab() -> bool:
    """Robust Colab detection."""
    return (
        ("COLAB_RELEASE_TAG" in os.environ)
        or ("COLAB_GPU" in os.environ)
        or os.path.exists("/content")
    )

COLAB_ENV = in_colab()


def mount_google_drive() -> bool:
    """Mount Google Drive in Colab, otherwise no-op."""
    if not COLAB_ENV:
        print("  Not running in Colab environment - skipping Google Drive mount")
        print("   (Using local checkpoint storage)")
        return False
    try:
        # If already mounted, keep going
        if os.path.exists('/content/drive/MyDrive/'):
            print(" Google Drive already mounted")
            return True
        from google.colab import drive  # safe: only on Colab
        print(" Mounting Google Drive...")
        drive.mount('/content/drive')
        print(" Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"  Google Drive mounting failed: {e}")
        print("  Continuing with local checkpoint storage")
        return False


# Initialize Google Drive mounting BEFORE creating checkpoint directory
DRIVE_MOUNTED = mount_google_drive()


def create_checkpoint_directory(base_dir=None) -> str:
    """
    Create (and return) a checkpoint directory.
    Priority:
      1) explicit base_dir (if provided)
      2) Google Drive path in Colab
      3) local ./checkpoints
    """
    if base_dir is not None:
        checkpoint_dir = base_dir
    elif COLAB_ENV and DRIVE_MOUNTED:
        checkpoint_dir = '/content/drive/MyDrive/banglabert_checkpoints'
    else:
        checkpoint_dir = './checkpoints'

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f" Checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


# -----------------------------
# Checkpoint helpers
# -----------------------------
def _best_ckpt_path(checkpoint_dir: str, fold: int) -> str:
    """Fixed best-checkpoint filename per fold."""
    return os.path.join(checkpoint_dir, f'best_fold_{fold}.pth')


def _delete_old_pths_for_fold(checkpoint_dir: str, fold: int, keep_path: str | None = None):
    """
    Delete all .pth files for a given fold except `keep_path`.
    Keeps JSON metadata files intact.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    for fname in os.listdir(checkpoint_dir):
        if not fname.endswith('.pth'):
            continue
        # Match both old pattern and new best_ pattern
        if fname.startswith(f'checkpoint_fold_{fold}_epoch_') or fname == f'best_fold_{fold}.pth':
            full = os.path.join(checkpoint_dir, fname)
            if keep_path is not None and os.path.abspath(full) == os.path.abspath(keep_path):
                continue
            try:
                os.remove(full)
            except Exception as e:
                print(f"  Warning: could not delete {full}: {e}")


def save_checkpoint(epoch, model, optimizer, scheduler, fold, val_f1, train_loss, config_obj, checkpoint_dir):
    """
    Save only the latest/best .pth for this fold (overwriting/deleting older .pth files),
    but ALWAYS append a JSON metadata file per improved epoch to keep history.
    """
    checkpoint = {
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'fold': int(fold),
        'val_f1': float(val_f1),
        'train_loss': float(train_loss),
        'config': config_obj.__dict__ if hasattr(config_obj, '__dict__') else config_obj,
        'timestamp': str(np.datetime64('now'))
    }

    # Fixed best path per fold
    best_path = _best_ckpt_path(checkpoint_dir, fold)

    # First, delete any previous .pth files for this fold (will also remove old best if present)
    _delete_old_pths_for_fold(checkpoint_dir, fold, keep_path=None)

    # Now save the new best
    torch.save(checkpoint, best_path)
    print(f" Checkpoint saved (best for fold {fold}): {best_path}")

    # Also save a metadata JSON for this improved epoch (do NOT delete old JSONs)
    metadata = {
        'fold': int(fold),
        'epoch': int(epoch),
        'val_f1': float(val_f1),
        'train_loss': float(train_loss),
        'checkpoint_path': best_path,  # current best
        'timestamp': checkpoint['timestamp']
    }
    metadata_path = os.path.join(checkpoint_dir, f'metadata_fold_{fold}_epoch_{epoch}.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"  Warning: could not write metadata JSON {metadata_path}: {e}")

    return best_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training checkpoint and return state."""
    print(f" Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'fold': checkpoint['fold'],
        'val_f1': checkpoint['val_f1'],
        'train_loss': checkpoint['train_loss'],
        'config': checkpoint.get('config', None)
    }


def _find_latest_legacy_checkpoint(checkpoint_dir: str, fold: int):
    """Back-compat: if old per-epoch files exist, pick the highest epoch."""
    pattern = f'checkpoint_fold_{fold}_epoch_'
    best = None
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.startswith(pattern) and file.endswith('.pth'):
                try:
                    epoch = int(file.split('_')[-1].split('.')[0])
                    path = os.path.join(checkpoint_dir, file)
                    if (best is None) or (epoch > best[0]):
                        best = (epoch, path)
                except ValueError:
                    continue
    return best  # (epoch, path) or None


def find_latest_checkpoint(checkpoint_dir, fold):
    """
    Find the resume checkpoint for a fold.
    Preference:
      1) New scheme: best_fold_{fold}.pth
      2) Legacy scheme: highest epoch checkpoint_fold_{fold}_epoch_*.pth
    Returns (epoch, path) or (None, None)
    """
    best_path = _best_ckpt_path(checkpoint_dir, fold)
    if os.path.exists(best_path):
        # Read epoch from file contents for accuracy
        try:
            ckpt = torch.load(best_path, map_location='cpu')
            return int(ckpt.get('epoch', 0)), best_path
        except Exception:
            # Fallback: unknown epoch, still resume from best
            return None, best_path

    legacy = _find_latest_legacy_checkpoint(checkpoint_dir, fold)
    if legacy is not None:
        return legacy[0], legacy[1]
    return None, None


def should_resume_from_checkpoint(checkpoint_dir, fold, config_obj):
    """Decide whether to resume from an existing checkpoint."""
    latest_epoch, checkpoint_path = find_latest_checkpoint(checkpoint_dir, fold)

    if checkpoint_path and (latest_epoch is None or latest_epoch < config_obj.num_epochs):
        ep_info = "unknown" if latest_epoch is None else latest_epoch
        print(f" Found checkpoint for fold {fold}, epoch {ep_info}")
        print(f"   Resuming from epoch {(0 if latest_epoch is None else latest_epoch + 1)}")
        return True, checkpoint_path, latest_epoch
    elif checkpoint_path and latest_epoch is not None and latest_epoch >= config_obj.num_epochs:
        print(f" Fold {fold} already completed (epoch {latest_epoch})")
        return False, checkpoint_path, latest_epoch
    else:
        print(f" No checkpoint found for fold {fold}, starting fresh")
        return False, None, None


# -----------------------------
# Metrics & weighting
# -----------------------------
def calculate_class_weights(labels):
    """Calculate balanced class weights; safe for missing classes."""
    # For HateSpeech (binary: column 0 is 0/1)
    hate_true = labels[:, 0]
    hate_pos = np.sum(hate_true)
    hate_neg = len(labels) - hate_pos
    hate_weight = (hate_neg / max(hate_pos, 1)) if hate_pos > 0 else 1.0

    # For Emotion (multi-class: column 1 has class ids)
    emotion_true = labels[:, 1].astype(int)
    emotion_counts = np.bincount(emotion_true)
    safe_counts = np.where(emotion_counts == 0, 1, emotion_counts)
    emotion_weights = len(labels) / (len(safe_counts) * safe_counts)

    return {
        'hate_weight': torch.FloatTensor([hate_weight]),
        'emotion_weights': torch.FloatTensor(emotion_weights)
    }


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for binary (hate) + multiclass (emotion)."""
    # HateSpeech metrics (binary)
    hate_true = y_true[:, 0]
    hate_pred = (y_pred[:, 0] > 0.5).astype(int)

    hate_accuracy = accuracy_score(hate_true, hate_pred)
    hate_precision = precision_score(hate_true, hate_pred, average='binary', zero_division=0)
    hate_recall = recall_score(hate_true, hate_pred, average='binary', zero_division=0)
    hate_f1 = f1_score(hate_true, hate_pred, average='binary', zero_division=0)

    # Emotion metrics (multi-class)
    emotion_true = y_true[:, 1].astype(int)
    emotion_pred = np.argmax(y_pred[:, 1:], axis=1)

    emotion_accuracy = accuracy_score(emotion_true, emotion_pred)
    emotion_precision = precision_score(emotion_true, emotion_pred, average='macro', zero_division=0)
    emotion_recall = recall_score(emotion_true, emotion_pred, average='macro', zero_division=0)
    emotion_f1 = f1_score(emotion_true, emotion_pred, average='macro', zero_division=0)

    # Overall metrics
    overall_accuracy = (hate_accuracy + emotion_accuracy) / 2.0
    overall_f1 = (hate_f1 + emotion_f1) / 2.0

    return {
        'hate_accuracy': hate_accuracy,
        'hate_precision': hate_precision,
        'hate_recall': hate_recall,
        'hate_f1': hate_f1,
        'emotion_accuracy': emotion_accuracy,
        'emotion_precision': emotion_precision,
        'emotion_recall': emotion_recall,
        'emotion_f1': emotion_f1,
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1
    }


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        # If you intend to apply class weights inside the model, integrate there.
        # Here we assume the model already accounts for weights when provided in config.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())

    return total_loss / max(len(dataloader), 1)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += float(loss.item())

            # Apply sigmoid to HateSpeech logits and softmax to Emotion logits
            predictions = outputs['logits']
            hate_pred = torch.sigmoid(predictions[:, :1])
            emotion_pred = torch.softmax(predictions[:, 1:], dim=1)

            # Combine predictions
            combined_pred = torch.cat([hate_pred, emotion_pred], dim=1)
            all_predictions.extend(combined_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss
    return metrics


# -----------------------------
# K-fold training orchestration
# -----------------------------
def run_kfold_training(config_obj, comments, labels, tokenizer, device):
    mlflow.set_experiment(config_obj.mlflow_experiment_name)

    checkpoint_dir = create_checkpoint_directory(getattr(config_obj, "checkpoint_dir", None))

    run_name = f"{config_obj.author_name}_{config_obj.batch_size}_{config_obj.learning_rate}_{config_obj.num_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            'batch_size': config_obj.batch_size,
            'learning_rate': config_obj.learning_rate,
            'num_epochs': config_obj.num_epochs,
            'num_folds': config_obj.num_folds,
            'max_length': config_obj.max_length,
            'freeze_base': config_obj.freeze_base,
            'dropout': config_obj.dropout,
            'author_name': config_obj.author_name,
            'model_path': config_obj.model_path,
            'scheduler_type': getattr(config_obj, "scheduler_type", "linear"),
            'warmup_steps': getattr(config_obj, "warmup_steps", 0),
        })

        kfold_splits = data.prepare_kfold_splits(comments, labels, config_obj.num_folds)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"Fold {fold + 1}/{config_obj.num_folds}")
            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            class_weights = calculate_class_weights(train_labels)

            train_dataset = data.HateSpeechDataset(
                train_comments, train_labels, tokenizer,
                config_obj.max_length, augment=True
            )
            val_dataset = data.HateSpeechDataset(
                val_comments, val_labels, tokenizer,
                config_obj.max_length, augment=False
            )

            train_loader = DataLoader(train_dataset, batch_size=config_obj.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config_obj.batch_size, shuffle=False)

            model = BertMultiLabelClassifier(
                config_obj.model_path,
                4,  # 1 for HateSpeech + 3 for Emotion classes
                dropout=config_obj.dropout,
                multi_task=True,
                config=config_obj
            )

            if config_obj.freeze_base:
                freeze_base_layers(model)

            model.to(device)

            optimizer = AdamW(
                model.parameters(),
                lr=config_obj.learning_rate,
                weight_decay=0.01,
                eps=1e-8
            )

            total_steps = max(len(train_loader) * config_obj.num_epochs, 1)

            # Dynamic scheduler selection based on config
            if getattr(config_obj, "scheduler_type", "linear") == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(0.1 * total_steps),
                    num_training_steps=total_steps
                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=getattr(config_obj, "warmup_steps", 0),
                    num_training_steps=total_steps
                )

            resume_from_checkpoint, checkpoint_path, latest_epoch = should_resume_from_checkpoint(checkpoint_dir, fold, config_obj)

            if resume_from_checkpoint:
                checkpoint_state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
                start_epoch = int(checkpoint_state['epoch']) + 1
                best_f1 = float(checkpoint_state['val_f1'])
                best_metrics = None
            else:
                start_epoch = 0
                best_f1 = -1.0  # ensure the first improvement is captured
                best_metrics = None

            patience = getattr(config_obj, "early_stopping_patience", 3)
            patience_counter = 0

            for epoch in range(start_epoch, config_obj.num_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
                val_metrics = evaluate_model(model, val_loader, device)

                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_metrics['overall_f1']:.4f}")
                print(f"  Hate F1: {val_metrics['hate_f1']:.4f}, Emotion F1: {val_metrics['emotion_f1']:.4f}")

                if val_metrics['overall_f1'] > best_f1:
                    best_f1 = float(val_metrics['overall_f1'])
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                    save_checkpoint(epoch, model, optimizer, scheduler, fold, best_f1, train_loss, config_obj, checkpoint_dir)
                else:
                    patience_counter += 1

                # Log epoch metrics to MLflow
                mlflow.log_metrics({
                    f"fold_{fold}_epoch_{epoch}_train_loss": float(train_loss),
                    f"fold_{fold}_epoch_{epoch}_val_loss": float(val_metrics['loss']),
                    f"fold_{fold}_epoch_{epoch}_overall_f1": float(val_metrics['overall_f1']),
                    f"fold_{fold}_epoch_{epoch}_hate_f1": float(val_metrics['hate_f1']),
                    f"fold_{fold}_epoch_{epoch}_emotion_f1": float(val_metrics['emotion_f1'])
                })

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Safety: if no improvement happened, fall back to last val_metrics
            if best_metrics is None:
                best_metrics = val_metrics

            fold_results.append(best_metrics)
            print(f"Fold {fold+1} Best - Overall F1: {best_metrics['overall_f1']:.4f}")

            # Log best fold metrics
            mlflow.log_metrics({
                f"fold_{fold}_best_overall_f1": float(best_metrics['overall_f1']),
                f"fold_{fold}_best_hate_f1": float(best_metrics['hate_f1']),
                f"fold_{fold}_best_emotion_f1": float(best_metrics['emotion_f1'])
            })

        # Calculate and log average metrics
        avg_metrics = {}
        keys_to_avg = [k for k in fold_results[0].keys() if k != 'loss']
        for key in keys_to_avg:
            avg_metrics[f"avg_{key}"] = float(np.mean([result[key] for result in fold_results]))

        print(f"\nAverage Results across {config_obj.num_folds} folds:")
        print(f"Overall F1: {avg_metrics['avg_overall_f1']:.4f}")
        print(f"Hate F1: {avg_metrics['avg_hate_f1']:.4f}")
        print(f"Emotion F1: {avg_metrics['avg_emotion_f1']:.4f}")

        mlflow.log_metrics(avg_metrics)

        # Log final model artifact (last fold's model state)
        mlflow.pytorch.log_model(model, "model")


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    import config as cfg_mod

    # Load configuration
    config = cfg_mod.parse_arguments()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_path)

    # Load and preprocess data
    comments, labels = data.load_and_preprocess_data(config.dataset_path)

    # Run training
    run_kfold_training(config, comments, labels, tokenizer, device)
