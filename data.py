import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch
import re
import emoji

# Bengali hate speech dataset labels
LABEL_COLUMNS = ['HateSpeech', 'Emotion']

def clean_bangla_text(text):
    """Clean and preprocess Bangla text"""
    if not isinstance(text, str):
        return ""
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove English characters and numbers
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep Bangla punctuation
    text = re.sub(r'[^\u0980-\u09FF\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
    
    return text

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256, augment=False):
        self.texts = texts
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Clean text
        text = clean_bangla_text(text)
        
        # Simple augmentation: random word dropout (during training only)
        if self.augment and np.random.random() < 0.1:
            words = text.split()
            if len(words) > 3:
                # Remove 10% of words randomly
                num_to_remove = max(1, int(len(words) * 0.1))
                indices_to_remove = np.random.choice(len(words), num_to_remove, replace=False)
                words = [word for i, word in enumerate(words) if i not in indices_to_remove]
                text = ' '.join(words)

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
    
    # Emotion: 'sad' -> 0, 'angry' -> 1, 'happy' -> 2 (multi-class)
    df['Emotion'] = df['Emotion'].map({'sad': 0, 'angry': 1, 'happy': 2}).fillna(0)
    
    # Clean text
    df['Comments'] = df['Comments'].apply(clean_bangla_text)
    
    # Remove empty texts after cleaning
    df = df[df['Comments'].str.len() > 0]
    
    texts = df['Comments'].values
    labels = df[LABEL_COLUMNS].values
    
    print(f"Loaded {len(texts)} samples after preprocessing")
    print(f"Label distribution:")
    print(f"HateSpeech: {np.sum(labels[:, 0])} hate, {len(labels) - np.sum(labels[:, 0])} nonhate")
    print(f"Emotion: sad={np.sum(labels[:, 1] == 0)}, angry={np.sum(labels[:, 1] == 1)}, happy={np.sum(labels[:, 1] == 2)}")
    
    return texts, labels

def prepare_kfold_splits(texts, labels, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    return kfold.split(texts)