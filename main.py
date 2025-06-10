#!/usr/bin/env python3
"""
Toxic Comment Classification - Main Experiment Runner
Authors: Yash Suryavanshi and Rohit Roy Chowdhury

This script runs the complete pipeline for toxic comment classification
including baseline models, transformer models, and comprehensive evaluation.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from data_utils import load_data, preprocess_data, analyze_data, create_data_splits
from baseline_models import run_baseline_experiments
from transformer_models import run_transformer_experiments
from evaluation import generate_comprehensive_report

def setup_environment():
    """Set up the environment and check requirements"""
    print("=== SETTING UP ENVIRONMENT ===")
    
    # Set random seeds
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Check device
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check data files
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        print("Please download the Kaggle Toxic Comment Classification dataset")
        sys.exit(1)
    
    print("Environment setup complete!\n")

def run_data_analysis():
    """Run data analysis and visualization"""
    print("=== RUNNING DATA ANALYSIS ===")
    
    # Load and preprocess data
    train_df, test_df, test_labels_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
    
    # Analyze dataset
    analyze_data(train_df, save_plots=True)
    
    print("Data analysis complete!\n")
    return train_df, test_df

def run_experiments(args):
    """Run the main experiments"""
    results = {}
    
    if args.baseline or args.all:
        print("Running baseline experiments...")
        baseline_results = run_baseline_experiments()
        results['baseline'] = baseline_results
    
    if args.transformers or args.all:
        print("Running transformer experiments...")
        transformer_results = run_transformer_experiments()
        results['transformers'] = transformer_results
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Toxic Comment Classification Experiments")
    parser.add_argument('--baseline', action='store_true', help='Run baseline models only')
    parser.add_argument('--transformers', action='store_true', help='Run transformer models only')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--analysis-only', action='store_true', help='Run data analysis only')
    parser.add_argument('--no-analysis', action='store_true', help='Skip data analysis')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Default to running all if no specific option is chosen
    if not any([args.baseline, args.transformers, args.all, args.analysis_only, args.report]):
        args.all = True
    
    # Setup
    setup_environment()
    
    start_time = time.time()
    
    try:
        # Data analysis
        if not args.no_analysis:
            train_df, test_df = run_data_analysis()
        
        if args.analysis_only:
            print("Data analysis completed. Exiting...")
            return
        
        # Run experiments
        if args.baseline or args.transformers or args.all:
            results = run_experiments(args)
            print(f"\nExperiments completed in {time.time() - start_time:.2f} seconds")
        
        # Generate report
        if args.report or args.all:
            print("Generating comprehensive report...")
            generate_comprehensive_report()
    
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        raise
    
    finally:
        total_time = time.time() - start_time
        print(f"\n=== TOTAL EXECUTION TIME: {total_time:.2f} seconds ===")

if __name__ == "__main__":
    main() 