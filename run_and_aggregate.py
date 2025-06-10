#!/usr/bin/env python3
"""
Runs the final two failed models and aggregates their results with all
previously successful runs to generate a complete, comprehensive report.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Ensure the environment is set up correctly
user_site = os.path.expanduser("~/.local/lib/python3.9/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Now it's safe to import our project modules
from transformer_models import run_transformer_experiments, create_comprehensive_comparison
from config import MODELS_CONFIG, RESULTS_DIR

def run_final_models_and_aggregate():
    """
    Runs the final failed models, loads all results, and generates
    a comprehensive comparison of all five transformer models.
    """
    print("🎯 Starting Final Run & Aggregation Process")
    
    # 1. Define the models that still need to be run
    models_to_run = {
        'deberta': MODELS_CONFIG['deberta'],
        'electra': MODELS_CONFIG['electra']
    }
    
    # Temporarily update the main config to only run these models
    original_config = MODELS_CONFIG.copy()
    MODELS_CONFIG.clear()
    MODELS_CONFIG.update(models_to_run)
    
    print(f"🚀 Models to run: {', '.join(models_to_run.keys())}")
    
    # 2. Run the training for the final models
    # This will use the per-model fixes (mixed_precision, batch_size)
    new_results = run_transformer_experiments()
    
    # Restore the original config
    MODELS_CONFIG.clear()
    MODELS_CONFIG.update(original_config)
    print("\n✅ Original model configuration restored.")
    
    # 3. Load all existing results from the results directory
    print("\n🔍 Loading all existing and new results for final report...")
    all_results = new_results.copy()
    
    # List of models that should have results
    all_model_keys = ['bert', 'roberta', 'deberta', 'hatebert', 'electra']
    
    for model_key in all_model_keys:
        if model_key in all_results:
            print(f"  - Found new result for '{model_key}'")
            continue
        
        # If not in the new results, try to load from file
        model_name = original_config[model_key]['model_name'].replace('/', '_')
        metric_file = RESULTS_DIR / f"{model_name}_test_metrics.csv"
        
        if metric_file.exists():
            print(f"  - Found existing result for '{model_key}' at {metric_file}")
            # The create_comprehensive_comparison function expects a nested dict
            # So we create a dummy structure
            df = pd.read_csv(metric_file)
            all_results[model_key] = {
                'test_metrics': df.iloc[0].to_dict()
            }
        else:
            print(f"  - WARNING: No result file found for '{model_key}'")

    # 4. Generate the final, comprehensive comparison
    if len(all_results) > 1:
        print("\n📊 Generating final comprehensive report for all models...")
        create_comprehensive_comparison(all_results)
    else:
        print("\n⚠️ Not enough model results to generate a comparison.")
        
    print("\n🎉 Process Complete!")

if __name__ == "__main__":
    run_final_models_and_aggregate() 