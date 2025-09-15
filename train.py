# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)

import data
from model import BertMultiLabelClassifier, freeze_base_layers, unfreeze_all_layers
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import mlflow

def calculate_class_weights(labels):
    """Calculate balanced class weights"""
    # For HateSpeech (binary)
    hate_pos = np.sum(labels[:, 0])
    hate_neg = len(labels) - hate_pos
    hate_weight = hate_neg / hate_pos if hate_pos > 0 else 1.0
    
    # For Emotion (multi-class)
    emotion_counts = np.bincount(labels[:, 1].astype(int))
    emotion_weights = len(labels) / (len(emotion_counts) * emotion_counts)
    
    return {
        'hate_weight': torch.FloatTensor([hate_weight]),
        'emotion_weights': torch.FloatTensor(emotion_weights)
    }

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
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
    overall_accuracy = (hate_accuracy + emotion_accuracy) / 2
    overall_f1 = (hate_f1 + emotion_f1) / 2
    
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
            
            # Apply sigmoid to HateSpeech logits and softmax to Emotion logits
            predictions = outputs['logits']
            hate_pred = torch.sigmoid(predictions[:, :1])
            emotion_pred = torch.softmax(predictions[:, 1:], dim=1)
            
            # Combine predictions
            combined_pred = torch.cat([hate_pred, emotion_pred], dim=1)
            all_predictions.extend(combined_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss
    
    return metrics

def run_kfold_training(config, comments, labels, tokenizer, device):
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"{config.author_name}_{config.batch_size}_{config.learning_rate}_{config.num_epochs}"):
        # Log parameters
        mlflow.log_params({
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'num_folds': config.num_folds,
            'max_length': config.max_length,
            'freeze_base': config.freeze_base,
            'dropout': config.dropout,
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

            train_dataset = data.HateSpeechDataset(
                train_comments, train_labels, tokenizer, 
                config.max_length, augment=True
            )
            val_dataset = data.HateSpeechDataset(
                val_comments, val_labels, tokenizer, 
                config.max_length, augment=False
            )

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

            model = BertMultiLabelClassifier(
                config.model_path, 
                4,  # 1 for HateSpeech + 3 for Emotion classes
                dropout=config.dropout,
                multi_task=True,
                config=config
            )
            
            if config.freeze_base:
                freeze_base_layers(model)
            model.to(device)

            optimizer = AdamW(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=0.01, 
                eps=1e-8
            )
            
            total_steps = len(train_loader) * config.num_epochs
            
            # Dynamic scheduler selection based on config
            if config.scheduler_type == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=int(0.1 * total_steps), 
                    num_training_steps=total_steps
                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=config.warmup_steps, 
                    num_training_steps=total_steps
                )

            best_f1 = 0
            patience = config.early_stopping_patience  # Use config value
            patience_counter = 0

            for epoch in range(config.num_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
                val_metrics = evaluate_model(model, val_loader, device)
                
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_metrics['overall_f1']:.4f}")
                print(f"  Hate F1: {val_metrics['hate_f1']:.4f}, Emotion F1: {val_metrics['emotion_f1']:.4f}")

                if val_metrics['overall_f1'] > best_f1:
                    best_f1 = val_metrics['overall_f1']
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                # Log epoch metrics to MLflow
                mlflow.log_metrics({
                    f"fold_{fold}_epoch_{epoch}_train_loss": train_loss,
                    f"fold_{fold}_epoch_{epoch}_val_loss": val_metrics['loss'],
                    f"fold_{fold}_epoch_{epoch}_overall_f1": val_metrics['overall_f1'],
                    f"fold_{fold}_epoch_{epoch}_hate_f1": val_metrics['hate_f1'],
                    f"fold_{fold}_epoch_{epoch}_emotion_f1": val_metrics['emotion_f1']
                })

            fold_results.append(best_metrics)
            print(f"Fold {fold+1} Best - Overall F1: {best_metrics['overall_f1']:.4f}")
            
            # Log best fold metrics
            mlflow.log_metrics({
                f"fold_{fold}_best_overall_f1": best_metrics['overall_f1'],
                f"fold_{fold}_best_hate_f1": best_metrics['hate_f1'],
                f"fold_{fold}_best_emotion_f1": best_metrics['emotion_f1']
            })

        # Calculate and log average metrics
        avg_metrics = {}
        for key in fold_results[0].keys():
            if key != 'loss':
                avg_metrics[f"avg_{key}"] = np.mean([result[key] for result in fold_results])
        
        print(f"\nAverage Results across {config.num_folds} folds:")
        print(f"Overall F1: {avg_metrics['avg_overall_f1']:.4f}")
        print(f"Hate F1: {avg_metrics['avg_hate_f1']:.4f}")
        print(f"Emotion F1: {avg_metrics['avg_emotion_f1']:.4f}")
        
        mlflow.log_metrics(avg_metrics)

        # Log model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    import config
    from transformers import BertTokenizer
    
    # Load configuration
    config = config.parse_arguments()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    
    # Load and preprocess data
    comments, labels = data.load_and_preprocess_data(config.dataset_path)
    
    # Run training
    run_kfold_training(config, comments, labels, tokenizer, device)
