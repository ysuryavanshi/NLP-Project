import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_utils import load_data, preprocess_data, create_data_splits, create_dataloaders

class ToxicClassifier(nn.Module):
    """Transformer with a classification head"""
    
    def __init__(self, model_name: str, num_classes: int = 6, dropout: float = 0.3):
        super(ToxicClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        
        try:
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
            print(f"Loaded {model_name} with safetensors.")
        except Exception:
            print(f"Safetensors failed. Trying again to download {model_name}.")
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TransformerTrainer:
    """Handles training and evaluation for a transformer model"""
    
    def __init__(self, model, device, use_mixed_precision, lr=2e-5, warmup_steps=500):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_model_state = None
        
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        print(f"Training on {device}, Mixed Precision: {'ENABLED' if self.scaler else 'DISABLED'}")
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.calculate_metrics(all_labels, all_preds)
        self.history['val_loss'].append(total_loss / len(val_loader))
        self.history['val_f1'].append(metrics['f1_micro'])
        
        return total_loss / len(val_loader), metrics
    
    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'roc_auc_micro': roc_auc_score(y_true, y_pred_proba, average='micro'),
            'roc_auc_macro': roc_auc_score(y_true, y_pred_proba, average='macro')
        }
        
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, label in enumerate(TARGET_COLUMNS):
            metrics[f'f1_{label}'] = f1_per_class[i]
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs, patience, min_delta):
        """Main training loop with early stopping."""
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps=self.warmup_steps, 
                                                    num_training_steps=len(train_loader) * num_epochs)
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 Micro: {val_metrics['f1_micro']:.4f}")
            
            if val_metrics['f1_micro'] > self.best_val_f1 + min_delta:
                self.best_val_f1 = val_metrics['f1_micro']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
                print("✨ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Stopping early after {patience} epochs with no improvement.")
                    break
            
            scheduler.step()
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
    
    def plot_training_history(self, model_name):
        """Plot training and validation loss and F1 score."""
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        ax1.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['val_loss'], label='Validation Loss', color='orange')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history['val_f1'], label='Validation F1 Micro', color='green', linestyle='--')
        ax2.set_ylabel('F1 Score')
        ax2.tick_params(axis='y')
        
        fig.tight_layout()
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.title(f'{model_name} Training History')
        plt.savefig(PLOTS_DIR / f'{model_name.replace("/", "_")}_training_history.png', dpi=DPI)
        plt.show()

def train_and_evaluate_model(model_key):
    """Full pipeline for training and evaluating a single transformer model."""
    config = MODELS_CONFIG[model_key]
    model_name = config['model_name']
    print(f"\n--- Starting Experiment for: {model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    train_split, val_split, test_split = create_data_splits(train_df, test_df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split, tokenizer, 
        batch_size=config['batch_size'], max_length=config['max_length']
    )

    try:
        classifier = ToxicClassifier(model_name)
        trainer = TransformerTrainer(
            classifier, DEVICE, config['mixed_precision'], 
            lr=config['learning_rate'], warmup_steps=config['warmup_steps']
        )

        trainer.train(train_loader, val_loader, config['epochs'],
                      config['early_stopping_patience'], config['early_stopping_min_delta'])
        
        trainer.plot_training_history(model_name)
        
        print("\n--- Evaluating on Test Set ---")
        _, test_metrics = trainer.evaluate(test_loader)
        print(f"Test F1 Micro: {test_metrics['f1_micro']:.4f}")

        # Save results
        pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / f'{model_name.replace("/", "_")}_test_metrics.csv', index=False)
        torch.save(trainer.model.state_dict(), MODELS_DIR / f'{model_name.replace("/", "_")}.pt')
        
        return {'test_metrics': test_metrics, 'history': trainer.history}

    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return None

def create_comprehensive_comparison(results):
    """Create a comprehensive plot comparing all trained models."""
    print("Creating comprehensive model comparison...")
    
    data = []
    for model_key, result in results.items():
        metrics = result['test_metrics']
        data.append({
            'Model': model_key.upper(),
            'F1_Micro': metrics['f1_micro'],
            'F1_Macro': metrics['f1_macro'],
            'ROC_AUC': metrics['roc_auc_micro'],
        })
    df = pd.DataFrame(data).sort_values('F1_Micro', ascending=False)
    
    df.to_csv(RESULTS_DIR / 'all_models_comparison.csv', index=False)
    print(f"Comparison table saved to {RESULTS_DIR / 'all_models_comparison.csv'}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16)

    # F1 Scores
    df.plot(x='Model', y=['F1_Micro', 'F1_Macro'], kind='bar', ax=axes[0],
            title='F1 Score Comparison', rot=45)
    
    # ROC-AUC Scores
    df.plot(x='Model', y='ROC_AUC', kind='bar', ax=axes[1],
            title='ROC-AUC Score', rot=45, color='green')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'comprehensive_transformer_comparison.png', dpi=300)
    print("Comprehensive plot saved.")
    plt.show()

if __name__ == '__main__':
    all_results = {}
    model_keys_to_run = ['bert', 'roberta', 'hatebert', 'electra', 'deberta']
    
    for key in model_keys_to_run:
        if key in MODELS_CONFIG:
            result = train_and_evaluate_model(key)
            if result:
                all_results[key] = result
    
    if len(all_results) > 1:
        create_comprehensive_comparison(all_results)
    
    print("\n--- All Transformer Experiments Finished ---") 