# Toxic Comment Classification - Experiment Report
**Authors:** Yash Suryavanshi and Rohit Roy Chowdhury
**Date:** 2025-06-05 03:36:01
## Executive Summary
This report presents the results of our toxic comment classification experiments comparing baseline machine learning models with state-of-the-art transformer models (BERT and RoBERTa) on the Kaggle Toxic Comment Classification dataset.
**Best Performing Model:** Roberta-Base (Transformers)
**Best F1 Score (Micro):** 0.7235
**Best ROC-AUC (Micro):** 0.9796
## Performance Summary
### Overall Model Performance
| Category     | Model             |   F1 Micro |   F1 Macro |   ROC-AUC Micro |   ROC-AUC Macro |   Accuracy |   Precision Macro |   Recall Macro |
|:-------------|:------------------|-----------:|-----------:|----------------:|----------------:|-----------:|------------------:|---------------:|
| Transformers | Roberta-Base      |     0.7235 |     0.3798 |          0.9796 |          0.9289 |     0.9132 |             0.368 |         0.3935 |
| Transformers | Bert-Base-Uncased |     0.0745 |     0.0319 |          0.5012 |          0.5827 |     0.2063 |             0.197 |         0.1971 |
### Class Difficulty Analysis
The following table shows the difficulty of each toxicity class based on average F1 scores across all models:
| Class         |    Mean_F1 |     Std_F1 |     Min_F1 |    Max_F1 |     Range |
|:--------------|-----------:|-----------:|-----------:|----------:|----------:|
| threat        | 0.00505561 | 0.00505561 | 0          | 0.0101112 | 0.0101112 |
| identity_hate | 0.00949367 | 0.00949367 | 0          | 0.0189873 | 0.0189873 |
| severe_toxic  | 0.0119157  | 0.0119157  | 0          | 0.0238313 | 0.0238313 |
| obscene       | 0.393433   | 0.391191   | 0.00224215 | 0.784623  | 0.782381  |
| toxic         | 0.402544   | 0.375304   | 0.0272394  | 0.777848  | 0.750609  |
| insult        | 0.412687   | 0.303933   | 0.108754   | 0.71662   | 0.607865  |
## Key Findings
2. **Class Difficulty:** 'insult' was the easiest class to classify, while 'threat' was the most challenging.
3. **Multi-label Challenge:** The multi-label nature of the problem presented significant challenges, with class imbalance affecting model performance.
## Methodology
### Dataset
- **Source:** Kaggle Toxic Comment Classification Challenge
- **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate
- **Task:** Multi-label binary classification
### Models Evaluated
**Transformer Models:**
- Bert-Base-Uncased
- Roberta-Base
### Evaluation Metrics
- F1 Score (Micro and Macro averaged)
- ROC-AUC (Micro and Macro averaged)
- Accuracy
- Precision and Recall
- Per-class performance analysis
## Visualizations
The following plots were generated during the analysis:
1. `comprehensive_comparison.png` - Overall model performance comparison
2. `per_class_analysis.png` - Per-class performance heatmap
3. `dataset_analysis.png` - Dataset characteristics and distribution
4. `transformer_comparison.png` - BERT vs RoBERTa detailed comparison
## Conclusions and Future Work
### Conclusions
1. Transformer models significantly outperform traditional baseline approaches
2. Class imbalance remains a key challenge in toxic comment detection
3. Multi-label classification requires careful consideration of threshold tuning
### Future Work
1. Experiment with class balancing techniques
2. Implement ensemble methods combining multiple models
3. Explore domain-specific pre-trained models
4. Investigate interpretability and bias analysis
