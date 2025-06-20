import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import re
import string
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from config import *

class ToxicCommentDataset(Dataset):
    """PyTorch dataset for toxic comments"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
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
            'labels': torch.FloatTensor(self.labels[idx])
        }

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test data from CSVs"""
    print("Loading data...")
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    test_labels_df = pd.read_csv(TEST_LABELS_FILE)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Test labels shape: {test_labels_df.shape}")
    
    return train_df, test_df, test_labels_df

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    
    return text.strip()

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, test_labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply cleaning and merge test labels"""
    print("Preprocessing data...")
    
    train_df['comment_text_clean'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text_clean'] = test_df['comment_text'].apply(clean_text)
    
    # Merge test data with labels, removing samples with -1 labels (unlabeled)
    test_df = test_df.merge(test_labels_df, on='id', how='left')
    mask = (test_df[TARGET_COLUMNS] != -1).all(axis=1)
    test_df = test_df[mask].reset_index(drop=True)
    
    print(f"After preprocessing - Train: {train_df.shape}, Test: {test_df.shape}")
    
    return train_df, test_df

def analyze_data(train_df: pd.DataFrame, save_plots: bool = True):
    """Create and save plots for dataset analysis"""
    print("Analyzing dataset...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Toxic Comment Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Label distribution
    label_counts = train_df[TARGET_COLUMNS].sum().sort_values(ascending=False)
    axes[0, 0].bar(range(len(label_counts)), label_counts.values)
    axes[0, 0].set_xticks(range(len(label_counts)))
    axes[0, 0].set_xticklabels(label_counts.index, rotation=45)
    axes[0, 0].set_title('Distribution of Toxicity Labels')
    axes[0, 0].set_ylabel('Count')
    
    for i, v in enumerate(label_counts.values):
        axes[0, 0].text(i, v + 500, str(v), ha='center', va='bottom')
    
    # Comment length distribution
    train_df['comment_length'] = train_df['comment_text_clean'].str.len()
    axes[0, 1].hist(train_df['comment_length'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Comment Lengths')
    axes[0, 1].set_xlabel('Character Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(train_df['comment_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {train_df["comment_length"].mean():.0f}')
    axes[0, 1].legend()
    
    # Correlation matrix of labels
    corr_matrix = train_df[TARGET_COLUMNS].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 0])
    axes[1, 0].set_title('Correlation Matrix of Toxicity Labels')
    
    # Multi-label statistics
    train_df['total_labels'] = train_df[TARGET_COLUMNS].sum(axis=1)
    label_dist = train_df['total_labels'].value_counts().sort_index()
    axes[1, 1].bar(label_dist.index, label_dist.values)
    axes[1, 1].set_title('Distribution of Number of Labels per Comment')
    axes[1, 1].set_xlabel('Number of Labels')
    axes[1, 1].set_ylabel('Count')
    
    total_comments = len(train_df)
    for i, v in enumerate(label_dist.values):
        percent = (v / total_comments) * 100
        axes[1, 1].text(label_dist.index[i], v + 1000, f'{percent:.1f}%', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(PLOTS_DIR / 'dataset_analysis.png', dpi=DPI, bbox_inches='tight')
        print(f"Dataset analysis plot saved to {PLOTS_DIR / 'dataset_analysis.png'}")
    
    plt.show()
    
    print("\n--- Dataset Summary ---")
    print(f"Total training samples: {len(train_df):,}")
    print(f"Avg comment length: {train_df['comment_length'].mean():.1f} chars")
    print(f"Clean comments: {((train_df[TARGET_COLUMNS].sum(axis=1) == 0).sum() / len(train_df) * 100):.2f}%")
    
    print("\n--- Label Stats ---")
    for col in TARGET_COLUMNS:
        count = train_df[col].sum()
        percentage = (count / len(train_df)) * 100
        print(f"{col}: {count:,} ({percentage:.2f}%)")

def create_data_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits, stratifying on the 'toxic' label"""
    print("Creating data splits...")
    
    train_split, val_split = train_test_split(
        train_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=train_df['toxic']
    )
    
    print(f"Train split: {len(train_split):,} samples")
    print(f"Validation split: {len(val_split):,} samples") 
    print(f"Test split: {len(test_df):,} samples")
    
    return train_split, val_split, test_df

def create_baseline_features(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, max_features: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create TF-IDF features for baseline models"""
    print("Creating TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.9
    )
    
    X_train = tfidf.fit_transform(train_df['comment_text_clean'])
    X_val = tfidf.transform(val_df['comment_text_clean'])
    X_test = tfidf.transform(test_df['comment_text_clean'])
    
    print(f"TF-IDF feature shape: {X_train.shape}")
    
    return X_train, X_val, X_test

def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                      tokenizer, batch_size: int = 16, max_length: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for transformers"""
    print("Creating DataLoaders...")
    
    train_dataset = ToxicCommentDataset(
        train_df['comment_text_clean'].tolist(),
        train_df[TARGET_COLUMNS].values,
        tokenizer,
        max_length
    )
    
    val_dataset = ToxicCommentDataset(
        val_df['comment_text_clean'].tolist(),
        val_df[TARGET_COLUMNS].values,
        tokenizer,
        max_length
    )
    
    test_dataset = ToxicCommentDataset(
        test_df['comment_text_clean'].tolist(),
        test_df[TARGET_COLUMNS].values,
        tokenizer,
        max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    analyze_data(train_df)
    train_split, val_split, test_split = create_data_splits(train_df, test_df) 