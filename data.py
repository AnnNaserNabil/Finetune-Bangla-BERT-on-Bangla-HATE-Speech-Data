import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch

# Bengali hate speech dataset labels
LABEL_COLUMNS = ['HateSpeech', 'Emotion']

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def load_and_preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)
    # Ensure label columns exist
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing label column: {col}")
    
    # Handle NaN values in labels - replace with most common value
    for col in LABEL_COLUMNS:
        most_common = df[col].mode()[0] if not df[col].empty else 'nonhate'
        df[col] = df[col].fillna(most_common)
    
    # Handle NaN values in text - replace with empty string
    df['Comments'] = df['Comments'].fillna('')
    
    # Convert categorical labels to numerical values
    # HateSpeech: 'hate' -> 1, 'nonhate' -> 0
    df['HateSpeech'] = df['HateSpeech'].map({'hate': 1, 'nonhate': 0}).fillna(0)
    
    # Emotion: 'sad' -> 1, 'angry' -> 0 (binary classification for simplicity)
    df['Emotion'] = df['Emotion'].map({'sad': 1, 'angry': 0}).fillna(0)
    
    texts = df['Comments'].values
    labels = df[LABEL_COLUMNS].values
    return texts, labels

def prepare_kfold_splits(texts, labels, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    return kfold.split(texts)