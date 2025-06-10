<!-- This report is auto-generated. Do not edit directly. -->

# Comprehensive Evaluation Report

This report summarizes the performance of all models evaluated for the toxic comment classification task.

## Performance Summary

| Category     | Model                             |   F1 Micro |   F1 Macro |   ROC-AUC Micro |   ROC-AUC Macro |   Accuracy |   Precision Macro |   Recall Macro |
|:-------------|:----------------------------------|-----------:|-----------:|----------------:|----------------:|-----------:|------------------:|---------------:|
| Transformers | Roberta-Base                      |     0.7235 |     0.3798 |          0.9796 |          0.9289 |     0.9132 |            0.368  |         0.3935 |
| Baseline     | Random Forest                     |     0.6993 |     0.4183 |          0.9681 |          0.9474 |     0.9138 |            0.7502 |         0.3678 |
| Baseline     | Logistic Regression               |     0.6477 |     0.4552 |          0.9763 |          0.9745 |     0.9173 |            0.7809 |         0.3457 |
| Transformers | Martin-Ha Toxic-Comment-Model     |     0.1473 |     0.0958 |          0.9353 |          0.8977 |     0.8893 |            0.2548 |         0.0775 |
| Transformers | Bert-Base-Uncased                 |     0.0745 |     0.0319 |          0.5012 |          0.5827 |     0.2063 |            0.197  |         0.1971 |
| Transformers | Google Electra-Base-Discriminator |     0.0576 |     0.0404 |          0.4185 |          0.5525 |     0.0002 |            0.0624 |         0.5169 |

## Performance Visualizations

The following charts compare the performance of all evaluated models.

![Comprehensive Model Comparison](plots/comprehensive_comparison.png)

![Per-Class Performance Analysis](plots/per_class_analysis.png)

## Confusion Matrices

Confusion matrices for the best performing transformer model.

*Prediction file not found at `/WAVE/users2/unix/ysuryavanshi/NLP Project/results/roberta-base_test_predictions.csv`, skipping confusion matrix generation.*

## Class Difficulty Analysis

This section analyzes which toxicity classes were the most challenging to predict across all models, based on average F1 scores.

| Class         |   Mean_F1 |    Std_F1 |     Min_F1 |   Max_F1 |    Range |
|:--------------|----------:|----------:|-----------:|---------:|---------:|
| threat        | 0.0531372 | 0.074146  | 0          | 0.196721 | 0.196721 |
| severe_toxic  | 0.0588101 | 0.0891479 | 0          | 0.254098 | 0.254098 |
| identity_hate | 0.0846928 | 0.0832155 | 0          | 0.237838 | 0.237838 |
| insult        | 0.385829  | 0.276367  | 0.0930041  | 0.71662  | 0.623616 |
| toxic         | 0.388434  | 0.349395  | 0.00736196 | 0.777848 | 0.770486 |
| obscene       | 0.450415  | 0.340309  | 0.00224215 | 0.810435 | 0.808193 |

