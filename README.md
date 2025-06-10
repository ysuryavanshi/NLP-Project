# Toxic Comment Classification

**Authors:** Yash Suryavanshi and Rohit Roy Chowdhury  
**Course:** NLP Project  
**Python Version:** 3.9.19

## Project Overview

This project implements a comprehensive multi-label toxic comment classification system using both traditional machine learning approaches and state-of-the-art transformer models. The goal is to automatically detect multiple types of toxicity in online comments to help moderators filter harmful content and improve community health.

## Dataset

- **Source:** [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Size:** ~159,000 Wikipedia talk comments
- **Task:** Multi-label binary classification
- **Labels:** 
  - `toxic` - General toxicity
  - `severe_toxic` - Severe toxicity
  - `obscene` - Obscene language
  - `threat` - Threats
  - `insult` - Insults
  - `identity_hate` - Identity-based hate

## Project Structure

```
NLP Project/
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed data files
├── models/                  # Saved model files
├── plots/                   # Generated visualizations
├── results/                 # Experimental results and metrics
├── nlp_env/                # Virtual environment
├── config.py               # Configuration and hyperparameters
├── data_utils.py           # Data loading and preprocessing
├── baseline_models.py      # Traditional ML models
├── transformer_models.py   # BERT and RoBERTa implementations
├── evaluation.py           # Comprehensive evaluation and reporting
├── main.py                 # Main experiment runner
├── demo.py                 # Quick demonstration script
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Models Implemented

### Baseline Models
1. **Logistic Regression** - Traditional linear classifier with TF-IDF features
2. **Random Forest** - Ensemble method with TF-IDF features

### Transformer Models
1. **BERT** (`bert-base-uncased`) - Bidirectional encoder representations
2. **RoBERTa** (`roberta-base`) - Optimized BERT variant

All models use One-vs-Rest strategy for multi-label classification.

## Features

- **Comprehensive Data Analysis:** Dataset visualization, label distribution, correlation analysis
- **Text Preprocessing:** Cleaning, normalization, and tokenization
- **Multiple Model Types:** Both traditional ML and deep learning approaches
- **Robust Evaluation:** Micro/macro averaged metrics, per-class analysis
- **Visualization:** Performance comparisons, training curves, confusion matrices
- **Automated Reporting:** Detailed markdown reports with findings

## Setup and Installation

### 1. Environment Setup

```bash
# Activate virtual environment
source nlp_env/bin/activate  # On Linux/Mac
# OR
nlp_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download

Download the dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place the files in `data/raw/`:

- `train.csv`
- `test.csv`
- `test_labels.csv`

### 3. Verify Setup

```bash
# Quick demo to verify everything works
python demo.py
```

## Usage

### Quick Demo

For a quick demonstration with a subset of data:

```bash
python demo.py
```

### Full Experiments

#### Run All Experiments
```bash
python main.py --all
```

#### Run Specific Experiments
```bash
# Only baseline models
python main.py --baseline

# Only transformer models
python main.py --transformers

# Only data analysis
python main.py --analysis-only

# Generate comprehensive report
python main.py --report
```

### Individual Components

#### Data Analysis
```python
from data_utils import load_data, preprocess_data, analyze_data

train_df, test_df, test_labels_df = load_data()
train_df, test_df = preprocess_data(train_df, test_df, test_labels_df)
analyze_data(train_df)
```

#### Baseline Models
```python
from baseline_models import run_baseline_experiments

results = run_baseline_experiments()
```

#### Transformer Models
```python
from transformer_models import run_transformer_experiments

results = run_transformer_experiments()
```

## Configuration

Key parameters can be modified in `config.py`:

```python
# Model configurations
MODELS_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3,
        'warmup_steps': 1000
    },
    'roberta': {
        'model_name': 'roberta-base',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3,
        'warmup_steps': 1000
    }
}

# Target columns for classification
TARGET_COLUMNS = [
    'toxic', 'severe_toxic', 'obscene', 
    'threat', 'insult', 'identity_hate'
]
```

## Evaluation Metrics

The project uses comprehensive evaluation metrics:

- **F1 Score** (Micro and Macro averaged)
- **ROC-AUC** (Micro and Macro averaged)  
- **Accuracy**
- **Precision and Recall** (Micro and Macro averaged)
- **Per-class Performance Analysis**

## Expected Results

Based on the project design, expected performance ranges:

| Model | F1 Micro | F1 Macro | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.85-0.90 | 0.60-0.70 | 0.85-0.92 |
| Random Forest | 0.83-0.88 | 0.58-0.68 | 0.83-0.90 |
| BERT | 0.90-0.95 | 0.70-0.80 | 0.92-0.97 |
| RoBERTa | 0.91-0.96 | 0.72-0.82 | 0.93-0.98 |

## Outputs

After running experiments, check these directories:

- `results/` - CSV files with detailed metrics, performance summaries
- `plots/` - Visualizations and comparison charts
- `models/` - Saved model checkpoints
- `results/experiment_report.md` - Comprehensive markdown report

## Hardware Requirements

### Minimum
- RAM: 8GB
- Storage: 5GB free space
- CPU: Multi-core processor

### Recommended
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (for transformer models)
- Storage: 10GB+ free space

## Timeline

Following the project proposal timeline:

1. **Week 1 (May 16-22):** Data loading, cleaning, baseline implementation ✅
2. **Week 2 (May 23-30):** Fine-tune BERT and RoBERTa; hyperparameter tuning
3. **Week 3 (May 31-June 5):** Evaluate models; generate plots; draft slides
4. **June 6:** Presentation
5. **Week 4 (June 7-12):** Write and finalize the report

## Common Issues and Solutions

### 1. CUDA Out of Memory
```python
# Reduce batch size in config.py
MODELS_CONFIG['bert']['batch_size'] = 8
MODELS_CONFIG['roberta']['batch_size'] = 8
```

### 2. Dataset Not Found
```bash
# Ensure files are in correct location
ls data/raw/
# Should show: train.csv, test.csv, test_labels.csv
```

### 3. Package Installation Issues
```bash
# Update pip first
pip install --upgrade pip

# Install with specific versions
pip install torch==1.12.0 transformers==4.21.0
```

## Contributing

This is an academic project. For questions or issues:

1. Check the project documentation
2. Review the configuration settings
3. Ensure dataset is properly downloaded
4. Verify environment setup

## References

1. Kaggle Toxic Comment Classification Challenge (2018)
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
3. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (arXiv 2019)
4. Wolf et al., "Transformers: State-of-the-Art NLP" (EMNLP 2020)
5. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (NeurIPS 2019)
6. Pedregosa et al., "Scikit-Learn: Machine Learning in Python" (JMLR 2011)

## License

This project is for academic purposes only. Please respect the Kaggle competition terms and conditions for dataset usage.

---

**Note:** This project implements a complete pipeline for toxic comment classification as part of an NLP course project. The implementation focuses on comparing traditional ML approaches with modern transformer models while providing comprehensive analysis and evaluation. 