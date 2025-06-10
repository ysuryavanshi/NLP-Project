#!/bin/bash
#SBATCH --job-name=nlp-test-fixes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ysuryavanshi@ucdavis.edu
#SBATCH --output=nlp-test-fixes-%j.out
#SBATCH --error=nlp-test-fixes-%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: ${SLURM_SUBMIT_DIR}"
echo "Configuration: TESTING FIXES - BERT + RoBERTa only"
echo "Resource allocation: $SLURM_CPUS_PER_TASK CPUs, $SLURM_MEM_PER_NODE memory total"
echo "Training: BERT and RoBERTa models with FIXES"
echo "Settings: 3 epochs each, batch size 128, testing bug fixes"
echo "Time allocation: 2 hours (for testing)"

# Change to job directory
cd "${SLURM_SUBMIT_DIR}"

# Load modules
echo "Loading TensorFlow module..."
module load TensorFlow/20231212

# Check GPU
echo "GPU Information:"
nvidia-smi

echo "Starting NLP Toxic Comment Classification with FIXES..."
echo "This will test the fixed BERT and RoBERTa models"
echo "Models: Logistic Regression, Random Forest, BERT, RoBERTa"

# Create a temporary config for testing only BERT and RoBERTa
python -c "
import config
# Backup original and create test config with only BERT and RoBERTa
models_config_backup = config.MODELS_CONFIG.copy()

# Only test BERT and RoBERTa with reduced settings for quick testing
config.MODELS_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'batch_size': 64,
        'max_length': 256,
        'epochs': 3,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'early_stopping_patience': 2,
        'early_stopping_min_delta': 0.001
    },
    'roberta': {
        'model_name': 'roberta-base', 
        'batch_size': 64,
        'max_length': 256,
        'epochs': 3,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'early_stopping_patience': 2,
        'early_stopping_min_delta': 0.001
    }
}

# Save test config
with open('config_test.py', 'w') as f:
    f.write(f'''
import torch
from pathlib import Path

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
PLOTS_DIR = Path('plots')

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration - TEST VERSION
MODELS_CONFIG = {config.MODELS_CONFIG}

# Training configuration
TARGET_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Visualization settings
DPI = 300
FIGSIZE = (12, 8)

# Memory optimization
MIXED_PRECISION = True
MAX_GRAD_NORM = 1.0
''')
"

echo "Command: python main.py --all"
echo "=========================================="
echo "STARTING TEST PIPELINE at $(date)"
echo "=========================================="

# Run with test config
PYTHONPATH=.:$PYTHONPATH python -c "
import sys
sys.path.insert(0, '.')
import config_test as config
sys.modules['config'] = config
from main import main
main(['--all'])
"

echo "Exit status: $?"
echo ""
echo "📁 RESULTS GENERATED:"
echo "   - Baseline models: Logistic Regression, Random Forest"
echo "   - Advanced models: BERT, RoBERTa (FIXED)"
echo "   - Test of all fixes implemented"
echo ""
echo "📊 Check results in:"
echo "   - results/ directory for all CSV files"
echo "   - plots/ directory for all visualizations"
echo "   - models/ directory for saved model checkpoints"

echo "=========================================="
echo "TEST PIPELINE FINISHED at $(date)"
echo "==========================================" 