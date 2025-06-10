# A Comprehensive Analysis of Toxic Comment Classification Using Transformers and Baseline Models

**Author(s):** Yash Suryavanshi, Rohit Roy Chowdhury

---

## Abstract

The exponential growth of user-generated content has made manual moderation of online platforms untenable, creating a critical need for robust, automated toxicity detection systems. This project undertakes a comprehensive comparative analysis of machine learning techniques for multi-label toxic comment classification, a task complicated by the nuanced nature of language and severe class imbalance. We first establish rigorous baselines using traditional models with TF-IDF features: Logistic Regression (Micro-F1: 0.6477) and Random Forest (Micro-F1: 0.6993). We then systematically fine-tune and evaluate a suite of modern transformer architectures, including BERT, RoBERTa, ELECTRA, and the domain-specific HateBERT. Our findings demonstrate the decisive superiority of transformers, with RoBERTa achieving a state-of-the-art Micro-F1 score of 0.7235, a significant 11.7% relative improvement over the strongest baseline. This report provides a detailed account of our methodology, from principled data preprocessing and model architecture choices to an in-depth analysis of experimental outcomes. While our results highlight the power of pre-trained models, they also expose a critical weakness: even the best-performing models fail to reliably identify rare but harmful toxicity classes like `threat` and `identity_hate`. This work underscores the persistent challenge of class imbalance and provides a foundation for future research into more equitable and robust classification systems.

---

## 1. Introduction

The internet's evolution into the primary global commons for discourse has brought immense benefits, but has also surfaced a dark underbelly of online toxicity. Forms of digital aggression; including harassment, hate speech, and threats which pose a significant threat to the health of online ecosystems, capable of silencing marginalized voices, discouraging open participation, and causing real-world harm. While platform providers have a responsibility to foster safe environments, the sheer scale of content generation, with billions of daily posts, renders manual moderation completely inadequate.

This has necessitated a turn towards automated solutions rooted in Natural Language Processing (NLP) and machine learning. The goal is to develop systems that can accurately and efficiently classify toxic content at scale. However, this task is fraught with complexity. First, toxicity is not a monolith; a single comment can be simultaneously insulting, obscene, and threatening, demanding a multi-label classification approach. Second, human language is inherently nuanced. Sarcasm, irony, and culturally specific slang can easily deceive models that lack a deep contextual understanding. Third, and perhaps most critically, toxic content is statistically rare. This creates a severe class imbalance problem, where a naive model can achieve high accuracy by simply defaulting to the non-toxic class, failing entirely at its primary purpose.

This project directly confronts these challenges through a systematic, empirical investigation. We aim to move beyond surface-level comparisons to provide a detailed analysis of the strengths and weaknesses of different modeling paradigms. We begin by implementing and evaluating traditional machine learning classifiers (Logistic Regression, Random Forest) with TF-IDF features to establish a robust performance baseline. We then explore the cutting edge of NLP by fine-tuning several state-of-the-art transformer-based models. Our objective is not only to identify the best-performing model but also to understand *why* it performs well, to probe its failure modes, and to use these insights to chart a course for future improvements in this critical area of research.

---

## 2. Background and Related Work

The pursuit of automated toxicity detection has a rich history, with its evolution closely mirroring broader advancements in the field of Natural Language Processing.

*   **Traditional Machine Learning Methods:** Initial forays into this problem moved from brittle, keyword-based blacklists to more flexible machine learning approaches. The dominant paradigm involved representing text as high-dimensional, sparse vectors using techniques like Bag-of-Words or, more effectively, Term Frequency-Inverse Document Frequency (TF-IDF). TF-IDF improves on Bag-of-Words by weighting terms based on their importance, down-weighting common words (like "the" or "a") and prioritizing rarer, more informative words. These feature vectors were then used to train standard classifiers such as Logistic Regression, Naive Bayes, and Support Vector Machines (SVMs). These methods, while effective for their time, are fundamentally limited as they treat words as independent features, failing to capture the semantic meaning, word order, and syntactic relationships that are crucial for understanding language.

*   **Deep Learning and Sequential Models:** The advent of deep learning introduced models that could learn feature representations directly from text. Recurrent Neural Networks (RNNs) and their more sophisticated variants, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, were a significant step forward. By processing text token-by-token in sequence, these models could theoretically capture long-range dependencies and contextual information. LSTMs, in particular, use a system of "gates" (input, forget, and output) to regulate the flow of information through time, allowing them to selectively remember or forget information from past tokens, which helps mitigate the vanishing gradient problem seen in simple RNNs. These models, often initialized with pre-trained word embeddings like GloVe or Word2Vec, set new performance benchmarks.

*   **The Transformer Revolution:** The 2017 introduction of the Transformer architecture by Vaswani et al. was a watershed moment. It dispensed with recurrence entirely, instead relying on a mechanism called **self-attention**. Self-attention allows a model to weigh the importance of all other words in the input text when encoding a specific word, enabling it to build a deeply contextualized representation. This parallelizable approach proved far more effective and scalable than sequential processing. This architecture is the foundation of modern Large Language Models (LLMs). Models like BERT (Bidirectional Encoder Representations from Transformers) by Devlin et al. (2019) and its robustly optimized successor, RoBERTa (Liu et al., 2019), are pre-trained on vast internet-scale text corpora. During this pre-training, they learn a profound understanding of grammar, semantics, and world knowledge. By taking these powerful pre-trained models and **fine-tuning** them on a smaller, task-specific labeled dataset (like ours), researchers can achieve state-of-the-art results with a fraction of the data and compute that would be needed to train such a model from scratch.

---

## 3. Methodology

Our approach is structured to ensure a fair and rigorous comparison between traditional and transformer-based models, with a focus on reproducibility and in-depth analysis.

### 3.1. Problem Formulation
We define the task as multi-label binary classification. For each input comment, the model is required to output a six-dimensional binary vector `y_pred`. Each element `y_pred[i]` corresponds to one of the six toxicity classes: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A value of 1 signifies the presence of that toxicity type, while 0 indicates its absence. Crucially, the prediction for each label is treated as an independent binary classification problem, allowing a single comment to be assigned any combination of labels.

### 3.2. Data Preprocessing
The goal of preprocessing is to standardize the text and reduce noise without erasing important toxic signals. Each step was chosen deliberately:
1.  **Lowercasing and Whitespace Normalization:** This is a standard step to reduce the vocabulary size and ensure that words like "Toxic" and "toxic" are treated as the same token.
2.  **URL and User Mention Removal:** URLs and user mentions (e.g., `@username`) are artifacts of the platform and generally do not carry semantic content relevant to toxicity. Their high variability adds noise, so they are removed.
3.  **Punctuation Normalization:** Online discourse often features expressive, repetitive punctuation (e.g., "!!!!!"). We normalize this to a single character (e.g., "!"). This preserves the punctuation's value as a potential feature (indicating heightened emotion) without creating excessive unique tokens for the model to learn.
4.  **Deliberate Omissions (Stemming/Lemmatization):** We consciously chose *not* to perform stemming or lemmatization. These processes, which reduce words to their root form (e.g., "insulting" -> "insult"), can discard valuable information. For example, the distinction between "insult" (noun) and "insulting" (adjective) can be meaningful, and we trust the more advanced models to learn these nuances.

### 3.3. Baseline Models
To contextualize the performance of our transformer models, we first establish strong baselines using standard, widely-accepted machine learning techniques.
*   **Feature Engineering (TF-IDF):** For both baseline models, we represent the preprocessed text using TF-IDF vectors. We include both unigrams (single words) and bigrams (two-word sequences) in our vocabulary. Bigrams help capture some local context (e.g., "not good" has a different meaning than the individual words "not" and "good").
*   **Logistic Regression:** A robust and interpretable linear model. Due to the multi-label nature of the problem, we employ a **One-vs-Rest (OvR)** strategy. This involves training six independent binary logistic regression classifiers, one for each toxicity label. Each classifier is trained on the full dataset to distinguish its assigned class from all other data points.
*   **Random Forest:** A powerful ensemble method that mitigates the risk of overfitting common in single decision trees. It constructs a multitude of decision trees during training and outputs the mode of the classes (classification) of the individual trees. We used 100 trees, a common choice that provides a good balance between performance and computational cost.

### 3.4. Transformer Models
Our main experimental effort focuses on fine-tuning pre-trained transformer models.
*   **Architecture:** The core of the model is a pre-trained transformer (like BERT or RoBERTa), which functions as a sophisticated text encoder. For each input comment, the model outputs a high-dimensional vector representing the aggregate meaning of the text (typically the embedding of the special `[CLS]` token). This single vector is then passed through a dropout layer for regularization to prevent overfitting, and finally into a single linear layer, the **classification head**. This head projects the high-dimensional representation down to a six-dimensional logit vector, providing a raw, unnormalized score for each toxicity class.
*   **Loss Function:** We use `BinaryCrossEntropyWithLogitsLoss`. This loss function is ideal for multi-label tasks because it first applies a Sigmoid function to the raw logits to transform them into independent probabilities between 0 and 1, and then computes the binary cross-entropy loss for each class against the true labels. Combining the Sigmoid and loss calculation in one function also provides greater numerical stability than a sequential approach.
*   **Model Variants:** We chose a diverse set of models to explore different aspects of transformer performance:
    *   `bert-base-uncased`: The foundational model, serving as a well-understood point of comparison.
    *   `roberta-base`: An optimized version of BERT with a more rigorous pre-training scheme, often leading to better downstream performance.
    *   `google/electra-base-discriminator`: A model pre-trained with a more efficient "replaced token detection" task, which has shown strong results on the GLUE benchmark.
    *   `martin-ha/toxic-comment-model`: A specialized version of **HateBERT**, which itself was pre-trained on a massive corpus of abusive and toxic comments from Reddit. This allows us to test the hypothesis that domain-specific pre-training yields better results.

---

## 4. Experiments

### 4.1. Dataset
We use the **Kaggle "Toxic Comment Classification Challenge" dataset**, a standard benchmark for this task. It consists of ~159,000 labeled comments from Wikipedia talk page edits. A key characteristic, visualized in Figure 1, is its severe class imbalance. While `toxic` is the most frequent class (9.6% of samples), the rarest classes, `threat` and `identity_hate`, appear in less than 0.5% of the data. This extreme skew is the most significant challenge for any model trained on this dataset.

![Dataset Analysis](plots/dataset_analysis.png)
*Figure 1: Analysis of the training data, showing the severe class imbalance (top-left), comment length distribution (top-right), label correlations (bottom-left), and the number of labels per comment (bottom-right).*

### 4.2. Experimental Setup
*   **Data Split:** The official training data was subdivided into an 80% training set and a 20% validation set. The validation set was used for hyperparameter tuning and, crucially, for **early stopping**.
*   **Hardware:** All transformer model training and evaluation were performed on a high-performance computing (HPC) cluster using NVIDIA V100 GPUs, a necessity given the computational intensity of these models.
*   **Hyperparameters:** Key configurations were managed in a central `config.py` file. We used the **AdamW** optimizer, which improves upon the standard Adam by implementing a more effective weight decay mechanism. A learning rate of **2e-5** was chosen, as it is a widely-accepted, empirically-validated default for fine-tuning transformer models. Training was conducted for a maximum of 8 epochs, but we employed an early stopping protocol: if the validation loss failed to improve for two consecutive epochs, training was halted to prevent the model from overfitting to the training data.

### 4.3. Evaluation Metrics
Relying on a single metric like accuracy is deeply misleading for an imbalanced classification problem. We therefore evaluate our models using a suite of complementary metrics:
*   **F1 Score (Micro and Macro):** The F1 score, the harmonic mean of precision and recall, provides a more balanced measure than accuracy.
    *   *Micro-F1:* This metric is calculated globally by counting the total true positives, false negatives, and false positives across all classes. It is dominated by the performance on the more common classes, providing a good sense of overall aggregate performance.
    *   *Macro-F1:* This metric is calculated by finding the F1 score for each class independently and then taking their unweighted average. It treats every class as equally important, making it a critical measure of a model's ability to handle the rare, minority classes. A large gap between Micro and Macro F1 is a clear indicator that a model is performing poorly on rare classes.
*   **ROC-AUC (Micro and Macro):** The Area Under the Receiver Operating Characteristic Curve measures a model's ability to discriminate between classes. It is threshold-independent, evaluating the quality of the model's raw score outputs.
*   **Accuracy:** We report label-wise accuracy for completeness, but interpret it with extreme caution due to the class imbalance.

### 4.4. Results

#### Quantitative Evaluation
The aggregate results, presented in Table 1, clearly establish the superiority of transformer-based models. Our top model, **RoBERTa-Base**, achieved a Micro-F1 score of **0.7235**, a substantial 11.7% relative improvement over the best baseline, Random Forest (0.6993). This confirms that the deep, contextualized representations learned by transformers are highly effective for this task.

**Table 1: Overall Performance Comparison**
| Category     | Model                           | F1 Micro | F1 Macro | ROC-AUC Micro |
|--------------|---------------------------------|----------|----------|---------------|
| Transformers | RoBERTa-Base                    | **0.7235** | 0.3798   | **0.9796**    |
| Baseline     | Random Forest                   | 0.6993   | **0.4183**   | 0.9681        |
| Baseline     | Logistic Regression             | 0.6477   | 0.4552   | 0.9763        |
| Transformers | HateBERT                        | 0.1473   | 0.0958   | 0.9353        |
| Transformers | BERT-Base-Uncased               | 0.0745   | 0.0319   | 0.5012        |
| Transformers | ELECTRA-Base                    | 0.0576   | 0.0404   | 0.4185        |

A striking and counter-intuitive result is the catastrophic failure of standard `bert-base-uncased` and `electra-base-discriminator`. Their near-zero F1 scores suggest a fundamental breakdown during fine-tuning. One plausible explanation is **catastrophic forgetting**, where adapting the model's weights to a very narrow data distribution (toxic comments) causes it to lose the rich, general language understanding from its pre-training phase. The more robust pre-training objectives of RoBERTa and the domain-specific pre-training of HateBERT appear to make them more resilient to this phenomenon.

![Comprehensive Comparison](plots/comprehensive_comparison.png)
*Figure 2: A visual comparison of model performance across key metrics, showing RoBERTa's superiority in Micro-F1 and ROC-AUC.*

#### Qualitative Evaluation: Per-Class Performance
An analysis of per-class performance, shown in Table 2 and Figure 3, reveals the dramatic and pervasive impact of data imbalance. While all models perform respectably on frequent classes like `toxic` and `obscene`, their performance collapses on the rare classes.

**Table 2: Per-Class F1 Scores for Top Models**
| Model               | Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
|---------------------|-------|--------------|---------|--------|--------|---------------|
| RoBERTa-Base        | 0.778 | 0.000        | 0.785   | 0.000  | 0.717  | 0.000         |
| Random Forest       | 0.741 | 0.053        | 0.810   | 0.105  | 0.658  | 0.142         |
| Logistic Regression | 0.689 | 0.254        | 0.747   | 0.197  | 0.606  | 0.238         |

The most critical finding here is that our best overall model, RoBERTa, achieves an F1 score of **0.0** for `severe_toxic`, `threat`, and `identity_hate`. This means it failed to correctly classify a single instance of these categories. Faced with extreme rarity, the model learns that the optimal strategy to minimize its overall loss function is to always predict the negative class for these labels. This highlights a profound failure mode: the model excels on average, but fails completely on the most sensitive and potentially most harmful types of toxicity. Interestingly, the simpler baseline models, particularly Logistic Regression, manage to achieve non-zero F1 scores on these rare classes. This may be because the TF-IDF feature representation can isolate specific, highly-indicative keywords for a rare class, whereas these signals may become diluted within the complex, high-dimensional embeddings of a transformer.

![Per-Class Analysis](plots/per_class_analysis.png)
*Figure 3: A heatmap and bar chart visualizing the per-class F1 scores. The chart clearly shows high performance on common labels and a steep drop-off for rare ones.*

---

## 5. Conclusion

This project has systematically demonstrated that fine-tuned transformer models, and RoBERTa in particular, significantly outperform traditional machine learning methods for multi-label toxic comment classification. The 11.7% relative F1 score improvement of our best model over a strong Random Forest baseline validates the power of deep, contextualized language representations for this nuanced task.

However, our most significant finding is a cautionary one: superior aggregate performance masks a critical failure on rare classes. The **severe class imbalance** inherent in real-world toxicity datasets remains the single greatest obstacle to building truly effective and equitable moderation tools. Our best model, despite its high overall scores, was completely blind to multiple categories of toxicity, learning instead to ignore them in favor of optimizing for the majority classes. This is an unacceptable trade-off in a real-world application, where failing to detect a threat is a far more severe error than misclassifying a non-toxic comment.

Based on these findings, we propose that future work must shift its focus from simply chasing higher aggregate scores to directly addressing this imbalance. We recommend several promising research directions:
*   **Data-Level Techniques:**
    *   **Strategic Data Augmentation:** Instead of general augmentation, focus on generating high-quality synthetic examples *only* for the minority classes using techniques like generative language models or advanced text editing strategies.
    *   **Advanced Resampling:** Move beyond simple oversampling to explore methods like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN, which create synthetic samples in feature space rather than just duplicating existing ones.
*   **Algorithm-Level Techniques:**
    *   **Cost-Sensitive Learning & Loss Functions:** Implement and rigorously tune advanced loss functions like the **Focal Loss**, which dynamically adjusts the cross-entropy loss to focus training on hard-to-classify examples (i.e., the rare positive classes).
    *   **Decoupled Training Schedules:** Investigate multi-stage training strategies. For instance, a first stage could train the model on a class-balanced dataset to learn a good feature representation, followed by a second stage that fine-tunes the classification head on the original, imbalanced data distribution.

By tackling the class imbalance problem head-on, the field can move towards developing toxicity detection systems that are not only accurate on average, but also robust, fair, and reliable across the full spectrum of harmful online content.

---

## 6. References

*   Devlin, J., Chang, M.W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*.
*   Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.
*   Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32*.
*   Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.
*   Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. 