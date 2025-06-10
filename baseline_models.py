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
    """Wrapper for training and evaluating baseline ML models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train a Logistic Regression model."""
        print("Training Logistic Regression...")
        
        lr_model = OneVsRestClassifier(
            LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
            n_jobs=-1
        )
        lr_model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = {
            'predictions': lr_model.predict(X_val),
            'probabilities': lr_model.predict_proba(X_val)
        }
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train a Random Forest model."""
        print("Training Random Forest...")
        
        rf_model = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'predictions': rf_model.predict(X_val),
            'probabilities': rf_model.predict_proba(X_val)
        }
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate a comprehensive set of metrics for multi-label classification."""
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        try:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
        except ValueError:
            metrics['roc_auc_micro'] = 0.0
            metrics['roc_auc_macro'] = 0.0
        
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, label in enumerate(TARGET_COLUMNS):
            metrics[f'f1_{label}'] = f1_per_class[i]
        
        return metrics
    
    def evaluate_model(self, model_name: str, y_val, save_results: bool = True):
        """Evaluate a trained model and save results."""
        print(f"Evaluating {model_name}...")
        
        y_pred = self.results[model_name]['predictions']
        y_pred_proba = self.results[model_name]['probabilities']
        
        metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
        self.results[model_name]['metrics'] = metrics
        
        print(f"\n--- {model_name.upper()} RESULTS ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        
        if save_results:
            pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f'{model_name}_metrics.csv', index=False)
            joblib.dump(self.models[model_name], MODELS_DIR / f'{model_name}.pkl')
    
    def plot_comparison(self, save_plot: bool = True):
        """Plot a comparison of baseline model performance."""
        if len(self.results) < 2:
            print("Need at least 2 models to compare.")
            return
        
        models = list(self.results.keys())
        metrics_to_plot = ['f1_micro', 'f1_macro', 'roc_auc_micro', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline Models Comparison', fontsize=16)
        
        metric_values = {m: [self.results[model]['metrics'][m] for model in models] for m in metrics_to_plot}
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            axes[0, 0].bar(x + i*width, metric_values[metric], width, label=metric)
        
        axes[0, 0].set_title('Overall Metrics Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        
        model1_f1 = [self.results[models[0]]['metrics'][f'f1_{label}'] for label in TARGET_COLUMNS]
        model2_f1 = [self.results[models[1]]['metrics'][f'f1_{label}'] for label in TARGET_COLUMNS]
        
        axes[0, 1].bar(TARGET_COLUMNS, model1_f1)
        axes[0, 1].set_title(f'Per-Class F1 - {models[0]}')
        
        axes[1, 0].bar(TARGET_COLUMNS, model2_f1)
        axes[1, 0].set_title(f'Per-Class F1 - {models[1]}')
        
        x_labels = np.arange(len(TARGET_COLUMNS))
        axes[1, 1].bar(x_labels - 0.2, model1_f1, 0.4, label=models[0])
        axes[1, 1].bar(x_labels + 0.2, model2_f1, 0.4, label=models[1])
        axes[1, 1].set_title('Per-Class F1 Comparison')
        axes[1, 1].set_xticks(x_labels)
        axes[1, 1].set_xticklabels(TARGET_COLUMNS, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(PLOTS_DIR / 'baseline_comparison.png', dpi=DPI)
        plt.show()
    
    def save_predictions(self, X_test, y_test, test_df):
        """Save test set predictions and metrics for each model."""
        print("Saving test predictions...")
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            test_metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / f'{model_name}_test_metrics.csv', index=False)
            
            pred_df = test_df.copy()
            for i, col in enumerate(TARGET_COLUMNS):
                pred_df[f'pred_{col}'] = y_pred[:, i]
                pred_df[f'prob_{col}'] = y_pred_proba[:, i]
            pred_df.to_csv(RESULTS_DIR / f'{model_name}_test_predictions.csv', index=False)
            print(f"Saved predictions and metrics for {model_name}")

def run_baseline_experiments():
    """Run the full pipeline for baseline experiments."""
    print("--- Running Baseline Experiments ---")
    
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    
    train_split, val_split, test_split = create_data_splits(train_df, test_df)
    
    X_train, X_val, X_test = create_baseline_features(train_split, val_split, test_split)
    y_train = train_split[TARGET_COLUMNS].values
    y_val = val_split[TARGET_COLUMNS].values
    y_test = test_split[TARGET_COLUMNS].values
    
    baseline = BaselineModels()
    baseline.train_logistic_regression(X_train, y_train, X_val, y_val)
    baseline.train_random_forest(X_train, y_train, X_val, y_val)
    
    baseline.evaluate_model('logistic_regression', y_val)
    baseline.evaluate_model('random_forest', y_val)
    
    baseline.plot_comparison()
    baseline.save_predictions(X_test, y_test, test_split)
    
    print("\n--- Baseline Experiments Complete ---")
    return baseline.results

if __name__ == "__main__":
    baseline = run_baseline_experiments() 