import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import glob
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

def load_all_results():
    """Load all experimental results from saved files"""
    results = {}
    
    # Load baseline results - look for logistic regression and random forest files
    baseline_files = glob.glob(str(RESULTS_DIR / "logistic_regression*metrics.csv")) + \
                    glob.glob(str(RESULTS_DIR / "random_forest*metrics.csv"))
    if baseline_files:
        results['baseline'] = {}
        for file in baseline_files:
            model_name = Path(file).stem.replace('_metrics', '').replace('_test', '')
            results['baseline'][model_name] = pd.read_csv(file).iloc[0].to_dict()
    
    # Load transformer results
    transformer_files = glob.glob(str(RESULTS_DIR / "*bert*metrics.csv")) + \
                       glob.glob(str(RESULTS_DIR / "*roberta*metrics.csv")) + \
                       glob.glob(str(RESULTS_DIR / "*electra*metrics.csv")) + \
                       glob.glob(str(RESULTS_DIR / "*martin-ha*metrics.csv"))
    if transformer_files:
        results['transformers'] = {}
        for file in transformer_files:
            model_name = Path(file).stem.replace('_metrics', '').replace('_test', '')
            results['transformers'][model_name] = pd.read_csv(file).iloc[0].to_dict()
    
    return results

def create_performance_comparison(results, save_plot=True):
    """Create comprehensive performance comparison across all models"""
    if not results:
        print("No results found to compare")
        return
    
    # Prepare data for plotting
    all_models = []
    all_metrics = []
    
    for category, models in results.items():
        for model_name, metrics in models.items():
            all_models.append(f"{category}_{model_name}")
            all_metrics.append(metrics)
    
    if not all_models:
        print("No models found in results")
        return
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overall F1 Scores
    f1_micro_scores = [metrics['f1_micro'] for metrics in all_metrics]
    f1_macro_scores = [metrics['f1_macro'] for metrics in all_metrics]
    
    x = np.arange(len(all_models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, f1_micro_scores, width, label='F1 Micro', alpha=0.8)
    axes[0, 0].bar(x + width/2, f1_macro_scores, width, label='F1 Macro', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(all_models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC-AUC Scores
    roc_auc_micro = [metrics['roc_auc_micro'] for metrics in all_metrics]
    roc_auc_macro = [metrics['roc_auc_macro'] for metrics in all_metrics]
    
    axes[0, 1].bar(x - width/2, roc_auc_micro, width, label='ROC-AUC Micro', alpha=0.8)
    axes[0, 1].bar(x + width/2, roc_auc_macro, width, label='ROC-AUC Macro', alpha=0.8)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('ROC-AUC Score')
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy and Precision
    accuracy_scores = [metrics['accuracy'] for metrics in all_metrics]
    precision_macro = [metrics['precision_macro'] for metrics in all_metrics]
    
    axes[0, 2].bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
    axes[0, 2].bar(x + width/2, precision_macro, width, label='Precision Macro', alpha=0.8)
    axes[0, 2].set_xlabel('Models')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Accuracy vs Precision')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(all_models, rotation=45, ha='right')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4-6. Per-class F1 scores for best performing models
    # Find actual indices first
    baseline_indices = [i for i, model in enumerate(all_models) if 'baseline' in model]
    transformer_indices = [i for i, model in enumerate(all_models) if 'transformer' in model]
    
    # Only calculate argmax if we have models
    best_baseline_idx = None
    best_transformer_idx = None
    
    if baseline_indices:
        baseline_f1_scores = [all_metrics[i]['f1_micro'] for i in baseline_indices]
        best_baseline_idx = baseline_indices[np.argmax(baseline_f1_scores)]
    
    if transformer_indices:
        transformer_f1_scores = [all_metrics[i]['f1_micro'] for i in transformer_indices]
        best_transformer_idx = transformer_indices[np.argmax(transformer_f1_scores)]
    
    if baseline_indices and best_baseline_idx is not None:
        baseline_f1_scores = [all_metrics[best_baseline_idx][f'f1_{label}'] for label in TARGET_COLUMNS]
        
        axes[1, 0].bar(TARGET_COLUMNS, baseline_f1_scores, alpha=0.8)
        axes[1, 0].set_title(f'Best Baseline: {all_models[best_baseline_idx]}')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    if transformer_indices and best_transformer_idx is not None:
        transformer_f1_scores = [all_metrics[best_transformer_idx][f'f1_{label}'] for label in TARGET_COLUMNS]
        
        axes[1, 1].bar(TARGET_COLUMNS, transformer_f1_scores, alpha=0.8, color='orange')
        axes[1, 1].set_title(f'Best Transformer: {all_models[best_transformer_idx]}')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Compare best baseline vs best transformer per class
        if baseline_indices and best_baseline_idx is not None:
            x_labels = np.arange(len(TARGET_COLUMNS))
            width = 0.35
            
            axes[1, 2].bar(x_labels - width/2, baseline_f1_scores, width, 
                          label=all_models[best_baseline_idx], alpha=0.8)
            axes[1, 2].bar(x_labels + width/2, transformer_f1_scores, width, 
                          label=all_models[best_transformer_idx], alpha=0.8)
            axes[1, 2].set_xlabel('Toxicity Labels')
            axes[1, 2].set_ylabel('F1 Score')
            axes[1, 2].set_title('Best Models Comparison')
            axes[1, 2].set_xticks(x_labels)
            axes[1, 2].set_xticklabels(TARGET_COLUMNS, rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'comprehensive_comparison.png', dpi=DPI, bbox_inches='tight')
        print(f"Comprehensive comparison plot saved to {PLOTS_DIR / 'comprehensive_comparison.png'}")
    
    plt.show()

def create_performance_table(results):
    """Create a performance summary table"""
    if not results:
        return None
    
    # Prepare data for table
    table_data = []
    
    for category, models in results.items():
        for model_name, metrics in models.items():
            row = {
                'Category': category.title(),
                'Model': model_name.replace('_', ' ').title(),
                'F1 Micro': f"{metrics['f1_micro']:.4f}",
                'F1 Macro': f"{metrics['f1_macro']:.4f}",
                'ROC-AUC Micro': f"{metrics['roc_auc_micro']:.4f}",
                'ROC-AUC Macro': f"{metrics['roc_auc_macro']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision Macro': f"{metrics['precision_macro']:.4f}",
                'Recall Macro': f"{metrics['recall_macro']:.4f}"
            }
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by F1 Micro score (descending)
    df['F1_Micro_Numeric'] = df['F1 Micro'].astype(float)
    df = df.sort_values('F1_Micro_Numeric', ascending=False)
    df = df.drop('F1_Micro_Numeric', axis=1)
    
    # Save to CSV
    df.to_csv(RESULTS_DIR / 'performance_summary.csv', index=False)
    print(f"Performance summary saved to {RESULTS_DIR / 'performance_summary.csv'}")
    
    return df

def create_per_class_analysis(results, save_plot=True):
    """Create detailed per-class performance analysis"""
    if not results:
        return
    
    # Collect per-class metrics
    all_models = []
    per_class_data = {label: [] for label in TARGET_COLUMNS}
    
    for category, models in results.items():
        for model_name, metrics in models.items():
            model_display_name = f"{category}_{model_name}"
            all_models.append(model_display_name)
            
            for label in TARGET_COLUMNS:
                per_class_data[label].append(metrics[f'f1_{label}'])
    
    # Create heatmap
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    
    # Heatmap of F1 scores
    heatmap_data = []
    for model in all_models:
        model_scores = []
        for label in TARGET_COLUMNS:
            idx = all_models.index(model)
            model_scores.append(per_class_data[label][idx])
        heatmap_data.append(model_scores)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=all_models, columns=TARGET_COLUMNS)
    
    sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', center=0.5, 
                square=True, ax=axes[0], fmt='.3f')
    axes[0].set_title('F1 Scores Heatmap by Model and Class')
    axes[0].set_xlabel('Toxicity Labels')
    axes[0].set_ylabel('Models')
    
    # Average per-class performance
    avg_per_class = [np.mean(per_class_data[label]) for label in TARGET_COLUMNS]
    std_per_class = [np.std(per_class_data[label]) for label in TARGET_COLUMNS]
    
    axes[1].bar(TARGET_COLUMNS, avg_per_class, yerr=std_per_class, capsize=5, alpha=0.8)
    axes[1].set_title('Average F1 Score per Class (with std dev)')
    axes[1].set_xlabel('Toxicity Labels')
    axes[1].set_ylabel('Average F1 Score')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(avg_per_class, std_per_class)):
        axes[1].text(i, avg + std + 0.01, f'{avg:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(PLOTS_DIR / 'per_class_analysis.png', dpi=DPI, bbox_inches='tight')
        print(f"Per-class analysis plot saved to {PLOTS_DIR / 'per_class_analysis.png'}")
    
    plt.show()

def analyze_class_difficulty(results):
    """Analyze which classes are hardest to predict"""
    if not results:
        return None
    
    # Collect all F1 scores per class
    class_scores = {label: [] for label in TARGET_COLUMNS}
    
    for category, models in results.items():
        for model_name, metrics in models.items():
            for label in TARGET_COLUMNS:
                class_scores[label].append(metrics[f'f1_{label}'])
    
    # Calculate statistics
    class_stats = []
    for label in TARGET_COLUMNS:
        scores = class_scores[label]
        stats = {
            'Class': label,
            'Mean_F1': np.mean(scores),
            'Std_F1': np.std(scores),
            'Min_F1': np.min(scores),
            'Max_F1': np.max(scores),
            'Range': np.max(scores) - np.min(scores)
        }
        class_stats.append(stats)
    
    # Create DataFrame and sort by difficulty (lowest mean F1)
    difficulty_df = pd.DataFrame(class_stats)
    difficulty_df = difficulty_df.sort_values('Mean_F1')
    
    # Save results
    difficulty_df.to_csv(RESULTS_DIR / 'class_difficulty_analysis.csv', index=False)
    print(f"Class difficulty analysis saved to {RESULTS_DIR / 'class_difficulty_analysis.csv'}")
    
    # Print summary
    print("\n=== CLASS DIFFICULTY ANALYSIS ===")
    print("Classes ranked by difficulty (hardest first):")
    for _, row in difficulty_df.iterrows():
        print(f"{row['Class']}: Mean F1 = {row['Mean_F1']:.4f} (±{row['Std_F1']:.4f})")
    
    return difficulty_df

def plot_confusion_matrices(y_true, y_pred, model_name):
    """Plot confusion matrices for each label"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Confusion Matrices for {model_name}', fontsize=16, fontweight='bold')
    
    for i, label in enumerate(TARGET_COLUMNS):
        ax = axes[i // 3, i % 3]
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(label)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = PLOTS_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=DPI)
    print(f"Confusion matrices saved to {save_path}")
    plt.close()

def generate_detailed_report(results):
    """Generate a detailed markdown report"""
    report_content = "<!-- This report is auto-generated. Do not edit directly. -->\n\n"
    report_content += "# Comprehensive Evaluation Report\n\n"
    report_content += "This report summarizes the performance of all models evaluated for the toxic comment classification task.\n\n"

    # --- Performance Summary Table ---
    report_content += "## Performance Summary\n\n"
    performance_df = create_performance_table(results)
    if performance_df is not None:
        report_content += performance_df.to_markdown(index=False) + "\n\n"
    else:
        report_content += "*No performance data found.*\n\n"

    # --- Visualizations ---
    report_content += "## Performance Visualizations\n\n"
    report_content += "The following charts compare the performance of all evaluated models.\n\n"
    report_content += "![Comprehensive Model Comparison](plots/comprehensive_comparison.png)\n\n"
    report_content += "![Per-Class Performance Analysis](plots/per_class_analysis.png)\n\n"
    
    return report_content

def generate_comprehensive_report():
    """
    Generate a comprehensive report with all analyses, plots, and summaries.
    """
    print("=== GENERATING COMPREHENSIVE REPORT ===")
    
    # 1. Load all results
    all_results = load_all_results()
    if not all_results:
        print("No results found. Cannot generate report.")
        return

    # 2. Create comparison plots and tables
    create_performance_comparison(all_results, save_plot=True)
    create_per_class_analysis(all_results, save_plot=True)
    
    # 3. Generate the main part of the report
    report_content = generate_detailed_report(all_results)

    # 4. Add Confusion Matrices to the report
    report_content += "## Confusion Matrices\n\n"
    report_content += "Confusion matrices for the best performing transformer model.\n\n"
    
    best_transformer_model_name = "bert" # Fallback
    if 'transformers' in all_results and all_results['transformers']:
        # Find the model with the highest f1_micro score
        best_transformer_model_name = max(all_results['transformers'], key=lambda m: all_results['transformers'][m].get('f1_micro', 0))

    # Path to the prediction file
    pred_file = RESULTS_DIR / f'{best_transformer_model_name}_test_predictions.csv'
    
    if pred_file.exists():
        try:
            pred_df = pd.read_csv(pred_file)
            y_true = pred_df[TARGET_COLUMNS].values
            
            # Determine if predictions are probabilities or binary labels
            if f'prob_{TARGET_COLUMNS[0]}' in pred_df.columns:
                y_pred_proba = pred_df[[f'prob_{col}' for col in TARGET_COLUMNS]].values
                y_pred = (y_pred_proba > 0.5).astype(int)
            elif f'pred_{TARGET_COLUMNS[0]}' in pred_df.columns:
                y_pred = pred_df[[f'pred_{col}' for col in TARGET_COLUMNS]].values
            else:
                raise ValueError("Prediction columns not found in test predictions file.")

            plot_confusion_matrices(y_true, y_pred, best_transformer_model_name)
            report_content += f"![Confusion Matrices for {best_transformer_model_name}](plots/confusion_matrix_{best_transformer_model_name}.png)\n\n"
        except Exception as e:
            report_content += f"*Error generating confusion matrix: {e}*\n\n"
    else:
        report_content += f"*Prediction file not found at `{pred_file}`, skipping confusion matrix generation.*\n\n"

    # 5. Add Class Difficulty Analysis
    report_content += "## Class Difficulty Analysis\n\n"
    report_content += "This section analyzes which toxicity classes were the most challenging to predict across all models, based on average F1 scores.\n\n"
    difficulty_df = analyze_class_difficulty(all_results)
    if difficulty_df is not None:
        report_content += difficulty_df.to_markdown(index=False) + "\n\n"
    
    # 6. Save the final report
    report_path = RESULTS_DIR / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
        
    print(f"Comprehensive report saved to {report_path}")
    print("Report generation complete!")

if __name__ == "__main__":
    generate_comprehensive_report() 