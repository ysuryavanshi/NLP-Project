#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path
import torch
import numpy as np
import random

sys.path.append(str(Path(__file__).parent))

from config import *
from data_utils import load_data, preprocess_data, analyze_data
from baseline_models import run_baseline_experiments
from transformer_models import train_and_evaluate_model, create_comprehensive_comparison
from evaluation import generate_comprehensive_report

def setup_environment():
    """Set random seeds and check GPU."""
    print("--- Setting up environment ---")
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print(f"Using device: {DEVICE}")
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training data not found at {TRAIN_FILE}")
        sys.exit(1)
    
    print("Setup complete.\n")

def main():
    """Main script for experiments."""
    parser = argparse.ArgumentParser(description="Toxic Comment Classification Experiments")
    parser.add_argument('--baseline', action='store_true', help='Run baseline models')
    parser.add_argument('--transformers', action='store_true', help='Run transformer models')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--analysis-only', action='store_true', help='Run only data analysis')
    parser.add_argument('--report-only', action='store_true', help='Generate final report only')
    
    args = parser.parse_args()
    
    if not any([args.baseline, args.transformers, args.all, args.analysis_only, args.report_only]):
        args.all = True
    
    setup_environment()
    start_time = time.time()
    
    try:
        if args.analysis_only:
            print("--- Running Data Analysis ---")
            train_df, test_df, test_labels_df = load_data()
            train_df, _ = preprocess_data(train_df, test_df, test_labels_df)
            analyze_data(train_df)
            return

        if args.report_only:
            print("--- Generating Final Report ---")
            generate_comprehensive_report()
            return
        
        all_results = {}
        if args.baseline or args.all:
            print("--- Running Baseline Experiments ---")
            all_results.update(run_baseline_experiments())
        
        if args.transformers or args.all:
            print("--- Running Transformer Experiments ---")
            transformer_results = {}
            for key in MODELS_CONFIG:
                result = train_and_evaluate_model(key)
                if result:
                    transformer_results[key] = result
            all_results['transformers'] = transformer_results

        if all_results:
            print("\n--- Generating Final Report ---")
            generate_comprehensive_report()
    
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        raise
    
    finally:
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 