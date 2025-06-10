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
    """Load all model result CSVs."""
    results = {'baseline': {}, 'transformers': {}}
    
    # Baseline models
    baseline_patterns = ["logistic_regression*metrics.csv", "random_forest*metrics.csv"]
    for pattern in baseline_patterns:
        for file in glob.glob(str(RESULTS_DIR / pattern)):
            model_name = Path(file).stem.replace('_metrics', '').replace('_test', '')
            results['baseline'][model_name] = pd.read_csv(file).iloc[0].to_dict()
    
    # Transformer models
    transformer_patterns = ["*bert*metrics.csv", "*roberta*metrics.csv", "*electra*metrics.csv", "*martin-ha*metrics.csv"]
    for pattern in transformer_patterns:
        for file in glob.glob(str(RESULTS_DIR / pattern)):
            model_name = Path(file).stem.replace('_metrics', '').replace('_test', '')
            results['transformers'][model_name] = pd.read_csv(file).iloc[0].to_dict()
    
    return results

def create_performance_comparison(results, save_plot=True):
    """Plot a comparison of all models."""
    if not results:
        print("No results to compare.")
        return
    
    all_models, all_metrics = [], []
    for category, models in results.items():
        for model_name, metrics in models.items():
            all_models.append(f"{category}_{model_name}")
            all_metrics.append(metrics)
    
    if not all_models:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16)
    
    df = pd.DataFrame(all_metrics, index=all_models)
    
    # F1 Scores, ROC-AUC, and Accuracy
    df[['f1_micro', 'f1_macro']].plot(kind='bar', ax=axes[0, 0], title='F1 Scores', rot=45)
    df[['roc_auc_micro', 'roc_auc_macro']].plot(kind='bar', ax=axes[0, 1], title='ROC-AUC Scores', rot=45)
    df[['accuracy']].plot(kind='bar', ax=axes[0, 2], title='Accuracy', rot=45)

    # Best baseline vs. best transformer
    baseline_df = df[df.index.str.contains('baseline')]
    transformer_df = df[df.index.str.contains('transformer')]

    if not baseline_df.empty and not transformer_df.empty:
        best_baseline = baseline_df['f1_micro'].idxmax()
        best_transformer = transformer_df['f1_micro'].idxmax()

        baseline_f1_per_class = [baseline_df.loc[best_baseline, f'f1_{label}'] for label in TARGET_COLUMNS]
        transformer_f1_per_class = [transformer_df.loc[best_transformer, f'f1_{label}'] for label in TARGET_COLUMNS]

        comparison_data = pd.DataFrame({
            best_baseline: baseline_f1_per_class,
            best_transformer: transformer_f1_per_class
        }, index=TARGET_COLUMNS)
        
        comparison_data.plot(kind='bar', ax=axes[1, 0], title='Best Baseline vs. Best Transformer (Per-Class F1)', rot=45)
        axes[1,0].set_ylabel('F1 Score')

    # Per-class f1 heatmap
    f1_cols = [f'f1_{label}' for label in TARGET_COLUMNS]
    per_class_df = df[f1_cols]
    per_class_df.columns = TARGET_COLUMNS
    sns.heatmap(per_class_df, annot=True, cmap='viridis', ax=axes[1, 1], fmt='.3f')
    axes[1, 1].set_title('Per-Class F1 Score Heatmap')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plot:
        plt.savefig(PLOTS_DIR / 'comprehensive_comparison.png', dpi=DPI)
    plt.show()

def create_performance_table(results):
    """Create and save a performance summary table."""
    if not results: return None
    
    table_data = []
    for category, models in results.items():
        for model_name, metrics in models.items():
            row = {'Category': category.title(), 'Model': model_name.replace('_', ' ').title()}
            for key, val in metrics.items():
                if isinstance(val, float):
                    row[key.replace('_', ' ').title()] = f"{val:.4f}"
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('F1 Micro', ascending=False)
    df.to_csv(RESULTS_DIR / 'performance_summary.csv', index=False)
    
    print(f"Performance summary saved to {RESULTS_DIR / 'performance_summary.csv'}")
    return df

def create_per_class_analysis(results, save_plot=True):
    """Create per-class performance analysis plots."""
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
    """Analyze and rank classes by prediction difficulty."""
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
    """Plot confusion matrices for each label for a model."""
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
    """Generate a detailed markdown report of results."""
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
    report_content += "### Per-Class F1 Score Analysis\n"
    report_content += "![Per-Class Analysis](plots/per_class_analysis.png)\n\n"
    
    # --- Class Difficulty ---
    report_content += "## Class Difficulty Analysis\n\n"
    difficulty_df = analyze_class_difficulty(results)
    if difficulty_df is not None:
        report_content += difficulty_df.to_markdown(index=False) + "\n\n"
    else:
        report_content += "*No difficulty analysis data found.*\n\n"

    # --- Save Report ---
    report_file = RESULTS_DIR / 'evaluation_report.md'
    report_file.write_text(report_content)
    print(f"Comprehensive report saved to {report_file}")

def generate_comprehensive_report():
    """Generate a full report with all analyses."""
    print("\n--- Starting Comprehensive Report Generation ---")
    results = load_all_results()
    
    if not results['baseline'] and not results['transformers']:
        print("No results found. Exiting.")
        return
    
    print(f"Found results for {len(results.get('baseline', {}))} baseline models and {len(results.get('transformers', {}))} transformer models.")
    
    create_performance_table(results)
    create_performance_comparison(results)
    
    print("\n--- Evaluation Report Generation Complete ---")

if __name__ == '__main__':
    generate_comprehensive_report() 