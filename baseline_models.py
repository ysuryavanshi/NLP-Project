import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_utils import load_data, preprocess_data, create_data_splits, create_baseline_features

class BaselineModels:
    """Baseline models for toxic comment classification"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression with One-vs-Rest for multi-label classification"""
        print("Training Logistic Regression...")
        
        # Use OneVsRestClassifier for multi-label classification
        lr_model = OneVsRestClassifier(
            LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            n_jobs=-1
        )
        
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lr_model.predict(X_val)
        y_pred_proba = lr_model.predict_proba(X_val)
        
        # Store model and results
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': lr_model
        }
        
        return lr_model, y_pred, y_pred_proba
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest with One-vs-Rest for multi-label classification"""
        print("Training Random Forest...")
        
        rf_model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_val)
        y_pred_proba = rf_model.predict_proba(X_val)
        
        # Store model and results
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': rf_model
        }
        
        return rf_model, y_pred, y_pred_proba
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate comprehensive metrics for multi-label classification"""
        metrics = {}
        
        # Overall metrics (micro and macro averages)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # ROC-AUC (using probabilities)
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
    
    def evaluate_model(self, model_name: str, y_val, save_results: bool = True):
        """Evaluate a trained model"""
        print(f"Evaluating {model_name}...")
        
        y_pred = self.results[model_name]['predictions']
        y_pred_proba = self.results[model_name]['probabilities']
        
        metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store metrics
        self.results[model_name]['metrics'] = metrics
        
        # Print results
        print(f"\n=== {model_name.upper()} RESULTS ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"ROC-AUC (Micro): {metrics['roc_auc_micro']:.4f}")
        print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        
        print(f"\n=== Per-Class F1-Scores ===")
        for label in TARGET_COLUMNS:
            print(f"{label}: {metrics[f'f1_{label}']:.4f}")
        
        if save_results:
            # Save detailed results to CSV
            results_df = pd.DataFrame([metrics])
            results_df.to_csv(RESULTS_DIR / f'{model_name}_metrics.csv', index=False)
            
            # Save model
            joblib.dump(self.models[model_name], MODELS_DIR / f'{model_name}.pkl')
        
        return metrics
    
    def plot_comparison(self, save_plot: bool = True):
        """Plot comparison of baseline models"""
        if len(self.results) < 2:
            print("Need at least 2 models to compare")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics_to_plot = ['f1_micro', 'f1_macro', 'roc_auc_micro', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline Models Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall metrics comparison
        metric_values = []
        for metric in metrics_to_plot:
            values = [self.results[model]['metrics'][metric] for model in models]
            metric_values.append(values)
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            axes[0, 0].bar(x + i*width, metric_values[i], width, label=metric)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Metrics Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-class F1 scores for first model
        model1_f1_scores = [self.results[models[0]]['metrics'][f'f1_{label}'] for label in TARGET_COLUMNS]
        axes[0, 1].bar(TARGET_COLUMNS, model1_f1_scores)
        axes[0, 1].set_title(f'Per-Class F1 Scores - {models[0]}')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Per-class F1 scores for second model
        model2_f1_scores = [self.results[models[1]]['metrics'][f'f1_{label}'] for label in TARGET_COLUMNS]
        axes[1, 0].bar(TARGET_COLUMNS, model2_f1_scores)
        axes[1, 0].set_title(f'Per-Class F1 Scores - {models[1]}')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: F1 scores comparison per class
        x = np.arange(len(TARGET_COLUMNS))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, model1_f1_scores, width, label=models[0])
        axes[1, 1].bar(x + width/2, model2_f1_scores, width, label=models[1])
        axes[1, 1].set_xlabel('Toxicity Labels')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Per-Class F1 Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(TARGET_COLUMNS, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(PLOTS_DIR / 'baseline_comparison.png', dpi=DPI, bbox_inches='tight')
            print(f"Baseline comparison plot saved to {PLOTS_DIR / 'baseline_comparison.png'}")
        
        plt.show()
    
    def save_predictions(self, X_test, y_test, test_df):
        """Make predictions on test set and save results"""
        print("Making predictions on test set...")
        
        for model_name, model in self.models.items():
            print(f"Predicting with {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate test metrics
            test_metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            print(f"{model_name} Test F1 (Micro): {test_metrics['f1_micro']:.4f}")
            print(f"{model_name} Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
            
            # Save predictions
            pred_df = test_df.copy()
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[f'pred_{col}'] = y_pred[:, i]
                pred_df[f'prob_{col}'] = y_pred_proba[:, i]
            
            pred_df.to_csv(RESULTS_DIR / f'{model_name}_test_predictions.csv', index=False)
            
            # Save test metrics
            test_results = pd.DataFrame([test_metrics])
            test_results.to_csv(RESULTS_DIR / f'{model_name}_test_metrics.csv', index=False)

def run_baseline_experiments():
    """Run complete baseline experiments"""
    print("=== RUNNING BASELINE EXPERIMENTS ===\n")
    
    # Load and preprocess data
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    
    # Create data splits
    train_split, val_split, test_split = create_data_splits(train_df, test_df)
    
    # Create TF-IDF features
    X_train, X_val, X_test = create_baseline_features(train_split, val_split, test_split)
    
    # Get labels
    y_train = train_split[TARGET_COLUMNS].values
    y_val = val_split[TARGET_COLUMNS].values
    y_test = test_split[TARGET_COLUMNS].values
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Train models
    baseline.train_logistic_regression(X_train, y_train, X_val, y_val)
    baseline.train_random_forest(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    baseline.evaluate_model('logistic_regression', y_val)
    baseline.evaluate_model('random_forest', y_val)
    
    # Plot comparisons
    baseline.plot_comparison()
    
    # Test set predictions
    baseline.save_predictions(X_test, y_test, test_split)
    
    print("\n=== BASELINE EXPERIMENTS COMPLETED ===")
    return baseline

if __name__ == "__main__":
    baseline = run_baseline_experiments() 