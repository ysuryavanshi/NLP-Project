import os
import torch
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

# Make sure dirs exist
for dir_path in [PROCESSED_DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Dataset Config ---
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
TEST_LABELS_FILE = RAW_DATA_DIR / "test_labels.csv"

# Classification targets
TARGET_COLUMNS = [
    'toxic', 'severe_toxic', 'obscene', 
    'threat', 'insult', 'identity_hate'
]

# --- Model Hyperparameters ---
MODELS_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'epochs': 8,
        'warmup_steps': 500,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 0.001,
        'mixed_precision': True
    },
    'roberta': {
        'model_name': 'roberta-base',
        'max_length': 512,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'epochs': 8,
        'warmup_steps': 500,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 0.001,
        'mixed_precision': True
    },
    'deberta': {
        'model_name': 'microsoft/deberta-base',
        'max_length': 512,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'epochs': 8,
        'warmup_steps': 500,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 0.001,
        'mixed_precision': False
    },
    'hatebert': {
        'model_name': 'martin-ha/toxic-comment-model',
        'max_length': 512,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'epochs': 8,
        'warmup_steps': 500,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 0.001,
        'mixed_precision': True
    },
    'electra': {
        'model_name': 'google/electra-base-discriminator',
        'max_length': 512,
        'batch_size': 32,
        'learning_rate': 3e-5,
        'epochs': 8,
        'warmup_steps': 500,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 0.001,
        'mixed_precision': True
    }
}

# --- Training Settings ---
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0

# --- Plotting ---
PLOT_STYLE = 'seaborn-v0_8'
DPI = 300 