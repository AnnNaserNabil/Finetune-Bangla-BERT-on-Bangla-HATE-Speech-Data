# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)

import data
from model import BertMultiLabelClassifier, freeze_base_layers
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import mlflow

def calculate_class_weights(labels):
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    weights = neg_counts / pos_counts
    return torch.FloatTensor(weights)

def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        if class_weights is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
            loss = loss_fct(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss
    return metrics

def run_kfold_training(config, comments, labels, tokenizer, device):
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{config.author_name}_{config.batch}_{config.lr}_{config.epochs}"):
        # Log parameters
        mlflow.log_params({
            'batch_size': config.batch,
            'learning_rate': config.lr,
            'num_epochs': config.epochs,
            'num_folds': config.num_folds,
            'max_length': config.max_length,
            'freeze_base': config.freeze_base,
            'author_name': config.author_name,
            'model_path': config.model_path
        })

        kfold_splits = data.prepare_kfold_splits(comments, labels, config.num_folds)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"Fold {fold + 1}/{config.num_folds}")
            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            class_weights = calculate_class_weights(train_labels)

            train_dataset = data.HateSpeechDataset(train_comments, train_labels, tokenizer, config.max_length)
            val_dataset = data.HateSpeechDataset(val_comments, val_labels, tokenizer, config.max_length)

            train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False)

            model = BertMultiLabelClassifier(config.model_path, len(data.LABEL_COLUMNS))
            if config.freeze_base:
                freeze_base_layers(model)
            model.to(device)

            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, eps=1e-8)
            total_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

            best_f1 = 0
            patience = 5
            patience_counter = 0

            for epoch in range(config.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
                val_metrics = evaluate_model(model, val_loader, device)
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}")

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                # Log epoch metrics to MLflow (per fold per epoch)
                mlflow.log_metrics({f"fold_{fold}_epoch_{epoch}_train_loss": train_loss, **{f"fold_{fold}_epoch_{epoch}_{k}": v for k, v in val_metrics.items()}})

            fold_results.append(best_metrics)
            # Log best fold metrics
            mlflow.log_metrics({f"fold_{fold}_{k}": v for k, v in best_metrics.items()})

        # Calculate and log average metrics
        avg_metrics = {key: np.mean([result[key] for result in fold_results]) for key in fold_results[0].keys() if key != 'loss'}
        mlflow.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})

        # Optionally log model
        mlflow.pytorch.log_model(model, "model")
