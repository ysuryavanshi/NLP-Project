import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Set environment variables to handle torch.load security issues
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import transformers with error handling
try:
    import transformers
    # Disable the weights_only warning/error if possible
    if hasattr(transformers, 'logging'):
        transformers.logging.set_verbosity_error()
except ImportError:
    pass

from config import *
from data_utils import load_data, preprocess_data, create_data_splits, create_dataloaders

class ToxicClassifier(nn.Module):
    """Multi-label transformer classifier for toxic comment detection"""
    
    def __init__(self, model_name: str, num_classes: int = 6, dropout: float = 0.3):
        super(ToxicClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load transformer model with multiple fallback strategies
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Strategy 1: Try safetensors first
        try:
            print(f"Attempting to load {model_name} with safetensors...")
            self.transformer = AutoModel.from_pretrained(
                model_name, 
                config=self.config,
                use_safetensors=True
            )
            print(f"✅ Successfully loaded {model_name} with safetensors")
        except (OSError, ValueError, Exception) as e:
            print(f"Safetensors loading failed: {str(e)[:100]}...")
            
            # Strategy 2: Try with local_files_only=False and force_download=True
            try:
                print(f"Attempting to reload {model_name} with force download...")
                self.transformer = AutoModel.from_pretrained(
                    model_name, 
                    config=self.config,
                    torch_dtype=torch.float32,
                    local_files_only=False,
                    force_download=True,
                    resume_download=True
                )
                print(f"✅ Successfully loaded {model_name} with force download")
            except Exception as e2:
                print(f"Force download failed: {str(e2)[:100]}...")
                
                # Strategy 3: Try with minimum parameters
                try:
                    print(f"Attempting basic loading of {model_name}...")
                    self.transformer = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    print(f"✅ Successfully loaded {model_name} with basic method")
                except Exception as e3:
                    print(f"❌ All loading strategies failed for {model_name}")
                    print(f"Final error: {e3}")
                    raise e3
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier layer"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class TransformerTrainer:
    """Trainer for transformer models with early stopping and memory optimization"""
    
    def __init__(self, model, device, use_mixed_precision, learning_rate=2e-5, warmup_steps=1000):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        # Early stopping
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_model_state = None
        
        # Mixed precision is now controlled EXPLICITLY by the parameter
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        print(f"Training on device: {device}")
        if self.scaler:
            print("Mixed precision training: ENABLED")
        else:
            print("Mixed precision training: DISABLED for this model")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params:,}")
        
    def train_epoch(self, train_loader):
        """Train for one epoch with memory optimization"""
        self.model.train()
        total_loss = 0
        
        # Setup progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
            
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.sigmoid(logits)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(all_labels, all_predictions)
        
        self.val_losses.append(avg_loss)
        self.val_f1_scores.append(metrics['f1_micro'])
        
        return avg_loss, metrics, all_predictions, all_labels
    
    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """Calculate comprehensive metrics"""
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # ROC-AUC
        try:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
        except:
            metrics['roc_auc_micro'] = 0.0
            metrics['roc_auc_macro'] = 0.0
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, label in enumerate(TARGET_COLUMNS):
            metrics[f'precision_{label}'] = precision_per_class[i]
            metrics[f'recall_{label}'] = recall_per_class[i]
            metrics[f'f1_{label}'] = f1_per_class[i]
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=3):
        """Train model with early stopping"""
        # Setup learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Training for {num_epochs} epochs...")
        # Get patience from any model config (they should all be the same)
        patience = MODELS_CONFIG[list(MODELS_CONFIG.keys())[0]]['early_stopping_patience']
        print(f"Early stopping patience: {patience} epochs")
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)
            val_f1 = val_metrics['f1_micro']
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            # Update scheduler
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val F1 (Micro): {val_f1:.4f}")
            print(f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"Val ROC-AUC (Micro): {val_metrics['roc_auc_micro']:.4f}")
            
            # Early stopping check
            # Get parameters from any model config (they should all be the same)
            first_model_key = list(MODELS_CONFIG.keys())[0]
            min_delta = MODELS_CONFIG[first_model_key]['early_stopping_min_delta']
            patience = MODELS_CONFIG[first_model_key]['early_stopping_patience']
            
            if val_f1 > self.best_val_f1 + min_delta:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            
            # Save best model
                model_name = self.model.model_name.replace('/', '_')
                torch.save(self.model.state_dict(), MODELS_DIR / f"best_{model_name}.pt")
                print(f"New best model saved! F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{patience}")
                
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered! Best F1: {self.best_val_f1:.4f}")
                    break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining completed. Best F1: {self.best_val_f1:.4f} at epoch {epoch + 1 - self.patience_counter}")
        return self.best_val_f1
    
    def plot_training_history(self, save_plot=True):
        """Plot training history"""
        if not self.train_losses or not self.val_losses:
            print("No training history to plot")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Ensure all lists have the same length
        min_length = min(len(self.train_losses), len(self.val_losses), len(self.val_f1_scores))
        epochs = range(1, min_length + 1)
        
        # Plot losses
        axes[0].plot(epochs, self.train_losses[:min_length], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.val_losses[:min_length], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot F1 scores
        axes[1].plot(epochs, self.val_f1_scores[:min_length], label='Validation F1', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            model_name = self.model.model_name.replace('/', '_')
            plt.savefig(PLOTS_DIR / f'{model_name}_training_history.png', dpi=DPI, bbox_inches='tight')
            print(f"Training history plot saved to {PLOTS_DIR / f'{model_name}_training_history.png'}")
        
        plt.show()

def train_transformer_model(model_name, train_loader, val_loader, epochs=3, model_key=None):
    """Train a transformer model"""
    print(f"\n=== Training {model_name} ===")
    
    # Initialize model
    model = ToxicClassifier(model_name, num_classes=len(TARGET_COLUMNS))
    
    # Get configuration - use model_key if provided, otherwise try to infer
    config = None
    if model_key and model_key in MODELS_CONFIG:
        config = MODELS_CONFIG[model_key]
    else:
        # Fallback: try to match model_name to config
        for key, cfg in MODELS_CONFIG.items():
            if cfg['model_name'] == model_name:
                config = cfg
                break
    
    if config:
        learning_rate = config.get('learning_rate', 2e-5)
        warmup_steps = config.get('warmup_steps', 1000)
        use_mixed_precision = config.get('mixed_precision', True)
    else:
        # Default values if no config found
        learning_rate = 2e-5
        warmup_steps = 1000
        use_mixed_precision = True
        print(f"Warning: No config found for {model_name}, using defaults")
    
    # Initialize trainer with the explicit flag
    trainer = TransformerTrainer(
        model=model,
        device=DEVICE,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        use_mixed_precision=use_mixed_precision
    )
    
    # Train model
    best_f1 = trainer.train(train_loader, val_loader, epochs)
    
    # Plot training history
    trainer.plot_training_history()
    
    return trainer, best_f1

def evaluate_on_test_set(trainer, test_loader, model_name):
    """Evaluate trained model on test set"""
    print(f"\n=== Evaluating {model_name} on Test Set ===")
    
    # Load best model
    best_model_path = MODELS_DIR / f"best_{model_name.replace('/', '_')}.pt"
    if best_model_path.exists():
        trainer.model.load_state_dict(torch.load(best_model_path, map_location=trainer.device, weights_only=False))
        print("Loaded best model checkpoint")
    
    # Evaluate
    test_loss, test_metrics, test_predictions, test_labels = trainer.evaluate(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1 (Micro): {test_metrics['f1_micro']:.4f}")
    print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test ROC-AUC (Micro): {test_metrics['roc_auc_micro']:.4f}")
    
    print(f"\n=== Per-Class Test F1 Scores ===")
    for label in TARGET_COLUMNS:
        print(f"{label}: {test_metrics[f'f1_{label}']:.4f}")
    
    # Save test results
    test_results_df = pd.DataFrame([test_metrics])
    test_results_df.to_csv(RESULTS_DIR / f'{model_name.replace("/", "_")}_test_metrics.csv', index=False)
    
    return test_metrics, test_predictions, test_labels

def compare_models(bert_metrics, roberta_metrics, save_plot=True):
    """Compare BERT and RoBERTa results"""
    models = ['BERT', 'RoBERTa']
    metrics_to_compare = ['f1_micro', 'f1_macro', 'roc_auc_micro', 'accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BERT vs RoBERTa Comparison', fontsize=16, fontweight='bold')
    
    # Overall metrics comparison
    metric_values = []
    for metric in metrics_to_compare:
        values = [bert_metrics[metric], roberta_metrics[metric]]
        metric_values.append(values)
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_compare):
        axes[0, 0].bar(x + i*width, metric_values[i], width, label=metric)
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Overall Metrics Comparison')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-class F1 comparison
    bert_f1_scores = [bert_metrics[f'f1_{label}'] for label in TARGET_COLUMNS]
    roberta_f1_scores = [roberta_metrics[f'f1_{label}'] for label in TARGET_COLUMNS]
    
    x = np.arange(len(TARGET_COLUMNS))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, bert_f1_scores, width, label='BERT')
    axes[0, 1].bar(x + width/2, roberta_f1_scores, width, label='RoBERTa')
    axes[0, 1].set_xlabel('Toxicity Labels')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Per-Class F1 Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(TARGET_COLUMNS, rotation=45)
    axes[0, 1].legend()
    
    # BERT per-class scores
    axes[1, 0].bar(TARGET_COLUMNS, bert_f1_scores)
    axes[1, 0].set_title('BERT Per-Class F1 Scores')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # RoBERTa per-class scores
    axes[1, 1].bar(TARGET_COLUMNS, roberta_f1_scores)
    axes[1, 1].set_title('RoBERTa Per-Class F1 Scores')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'transformer_comparison.png', dpi=DPI, bbox_inches='tight')
        print(f"Transformer comparison plot saved to {PLOTS_DIR / 'transformer_comparison.png'}")
    
    plt.show()

def run_transformer_experiments():
    """Run complete transformer experiments"""
    print("=== RUNNING TRANSFORMER EXPERIMENTS ===\n")
    
    # Load and preprocess data
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    
    # Create data splits
    train_split, val_split, test_split = create_data_splits(train_df, test_df)
    
    # Results storage
    results = {}
    
    # Train ALL configured models
    model_keys = list(MODELS_CONFIG.keys())
    print(f"Training {len(model_keys)} models: {', '.join(model_keys)}")
    
    for i, model_key in enumerate(model_keys, 1):
        model_name = MODELS_CONFIG[model_key]['model_name']
        batch_size = MODELS_CONFIG[model_key]['batch_size']
        max_length = MODELS_CONFIG[model_key]['max_length']
        epochs = MODELS_CONFIG[model_key]['epochs']
        
        print(f"\n{'='*60}")
        print(f"Training Model {i}/{len(model_keys)}: {model_name}")
        print(f"Key: {model_key}")
        print(f"Batch Size: {batch_size}, Max Length: {max_length}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        try:
            # Create tokenizer and data loaders
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_loader, val_loader, test_loader = create_dataloaders(
                train_split, val_split, test_split, tokenizer, batch_size, max_length
            )
            
            # Train model
            trainer, best_f1 = train_transformer_model(model_name, train_loader, val_loader, epochs, model_key)
            
            # Evaluate on test set
            test_metrics, test_predictions, test_labels = evaluate_on_test_set(trainer, test_loader, model_name)
            
            results[model_key] = {
                'trainer': trainer,
                'test_metrics': test_metrics,
                'test_predictions': test_predictions,
                'test_labels': test_labels,
                'best_f1': best_f1
            }
            
            print(f"✅ {model_key.upper()} completed successfully!")
            print(f"   Best F1: {best_f1:.4f}")
            print(f"   Test F1-Micro: {test_metrics['f1_micro']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {model_key}: {str(e)}")
            print("Continuing with next model...")
            continue
    
    # Generate comprehensive comparison
    if len(results) > 1:
        print(f"\n=== GENERATING COMPREHENSIVE COMPARISON ===")
        create_comprehensive_comparison(results)
    
    print(f"\n=== TRANSFORMER EXPERIMENTS COMPLETED ===")
    print(f"Successfully trained {len(results)} out of {len(model_keys)} models")
    return results

def create_comprehensive_comparison(results):
    """Create comprehensive comparison of all trained models"""
    print("Creating comprehensive model comparison...")
    
    # Create comparison DataFrame
    comparison_data = []
    for model_key, result in results.items():
        metrics = result['test_metrics']
        comparison_data.append({
            'Model': model_key.upper(),
            'Model_Name': MODELS_CONFIG[model_key]['model_name'],
            'Test_F1_Micro': metrics['f1_micro'],
            'Test_F1_Macro': metrics['f1_macro'],
            'Test_ROC_AUC': metrics['roc_auc_micro'],
            'Test_Accuracy': metrics['accuracy'],
            'Precision': metrics['precision_micro'],
            'Recall': metrics['recall_micro']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_F1_Micro', ascending=False)
    
    # Save comparison table
    comparison_df.to_csv(RESULTS_DIR / 'all_models_comparison.csv', index=False)
    print(f"Comparison table saved to {RESULTS_DIR / 'all_models_comparison.csv'}")
    
    # Print ranking
    print(f"\n📊 MODEL PERFORMANCE RANKING:")
    print("-" * 70)
    for idx, row in comparison_df.iterrows():
        print(f"{idx+1:2d}. {row['Model']:<10} | F1-Micro: {row['Test_F1_Micro']:.4f} | "
              f"F1-Macro: {row['Test_F1_Macro']:.4f} | ROC-AUC: {row['Test_ROC_AUC']:.4f}")
    
    # Create visualization
    create_all_models_plot(comparison_df, results)

def create_all_models_plot(comparison_df, results):
    """Create comprehensive visualization of all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison - All Transformers', fontsize=16, fontweight='bold')
    
    models = comparison_df['Model'].values
    
    # 1. Overall metrics comparison
    metrics_to_plot = ['Test_F1_Micro', 'Test_F1_Macro', 'Test_ROC_AUC', 'Test_Accuracy']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].values
        axes[0, 0].bar(x + i*width, values, width, label=metric.replace('Test_', ''), alpha=0.8)
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Overall Performance Comparison')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1-Micro ranking
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = axes[0, 1].bar(models, comparison_df['Test_F1_Micro'], color=colors)
    axes[0, 1].set_title('F1-Micro Score Ranking')
    axes[0, 1].set_ylabel('F1-Micro Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Per-category heatmap
    category_data = []
    for model_key in [k.lower() for k in models]:
        if model_key in results:
            metrics = results[model_key]['test_metrics']
            category_scores = [metrics[f'f1_{label}'] for label in TARGET_COLUMNS]
            category_data.append(category_scores)
    
    if category_data:
        category_matrix = np.array(category_data)
        im = axes[1, 0].imshow(category_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[1, 0].set_xticks(range(len(TARGET_COLUMNS)))
        axes[1, 0].set_xticklabels(TARGET_COLUMNS, rotation=45)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels(models)
        axes[1, 0].set_title('Per-Category F1 Scores Heatmap')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(TARGET_COLUMNS)):
                if i < len(category_data):
                    text = axes[1, 0].text(j, i, f'{category_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 4. Best model per category
    if category_data:
        category_best = []
        category_scores = []
        for j, label in enumerate(TARGET_COLUMNS):
            best_scores = [category_matrix[i, j] for i in range(len(models))]
            best_idx = np.argmax(best_scores)
            category_best.append(models[best_idx])
            category_scores.append(best_scores[best_idx])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(TARGET_COLUMNS)))
        bars = axes[1, 1].bar(TARGET_COLUMNS, category_scores, color=colors)
        axes[1, 1].set_title('Best F1 Score per Category')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add model names on bars
        for i, (bar, model) in enumerate(zip(bars, category_best)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{model}\n{height:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'comprehensive_transformer_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved to {PLOTS_DIR / 'comprehensive_transformer_comparison.png'}")
    plt.show()

if __name__ == "__main__":
    results = run_transformer_experiments() 