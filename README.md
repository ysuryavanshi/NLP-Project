# Toxic Comment Classification

This project builds a system to classify toxic online comments using both traditional ML models and transformers. The goal is to identify different types of toxicity, like insults, threats, and hate speech.

## Dataset

The data comes from Kaggle's [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), which has about 160,000 comments from Wikipedia talk pages. Each comment is labeled for six types of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## Project Structure

```
.
├── data/
├── models/
├── plots/
├── results/
├── config.py
├── data_utils.py
├── baseline_models.py
├── transformer_models.py
├── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```

## Models

*   **Baseline:** Logistic Regression & Random Forest with TF-IDF features.
*   **Transformers:** Fine-tuned BERT, RoBERTa, and other variants.

All models are set up for multi-label classification.

## How to Run

### 1. Setup

First, make sure you have Python 3.9+ and then install the dependencies.

```bash
# It's a good idea to use a virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Get the Data

Download the dataset from the [Kaggle page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and put `train.csv`, `test.csv`, and `test_labels.csv` into the `data/raw/` directory.

### 3. Run Experiments

You can run everything with a single command:

```bash
python main.py --all
```

Or run specific parts:

```bash
# Just run the baseline models
python main.py --baseline

# Just run the transformer models
python main.py --transformers

# Just generate the final plots and summary CSVs
python main.py --report-only
```

## Configuration

You can change model hyperparameters, like learning rate and batch size, in `config.py`.

```python
MODELS_CONFIG = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 3,
    },
    # ... other models
}
```

## Results

After running, you can find the outputs here:

*   `results/`: All the raw numbers, including performance summaries and per-class metrics.
*   `plots/`: Visuals comparing the model performances.
*   `models/`: Saved model files for later use.

## Common Issues

*   **CUDA Out of Memory:** If you get this error while training transformers, try reducing the `batch_size` for that model in `config.py`.
*   **Dataset Not Found:** Make sure the CSV files from Kaggle are in the `data/raw/` folder.

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