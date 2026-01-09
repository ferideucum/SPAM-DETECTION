# SPAM DETECTION AND TEXT CLASSIFICATION PROJECT

This project was developed by students of Fatih Sultan Mehmet Vakıf University to perform binary text classification (Spam vs. Ham) using Machine Learning and Deep Learning techniques.


## 👥 Project Team
* **Feride Uçum** - 2221251022
* **Furkan Emre Karçıkar** - 2221251007
* **Semih Alev** - 2221251010

---

## 📝 Abstract
In this project, a robust system for binary text classification (Spam vs. Ham) was developed and analyzed. The study explores the transition from classical Machine Learning algorithms (Logistic Regression, Decision Trees) to modern Deep Learning architectures (Multi-Layer Perceptrons, Bi-LSTM).

A significant portion of the work focused on **Preprocessing**, **Feature Engineering** (extracting domain-specific features), and **Overfitting Prevention** mechanisms such as Early Stopping, Dropout, and L2 Regularization.

The results demonstrate that the **MLP model utilizing TF-IDF** achieved the highest accuracy (**93%**), outperforming complex sequential models for this specific dataset.

---

## 📊 Data Collection
The dataset was collected from YouTube video comments using Python-based web scraping scripts and the `youtube_comment_downloader` library.

* **Total Data:** 2339 Comments
* **Normal (Ham) Comments:** 1170 (Label: 0)
* **Spam Comments:** 1169 (Label: 1)
* **Source:** 15 Normal video URLs, 42 Spam video URLs.

**Collection Strategy:**
* **Normal Comments:** Filtered for English only, checked against spam-like signals (links, contact info), and deduplicated.
* **Spam Comments:** Targeted specific spam patterns (URLs, CTA), with "URL-only" spam capped at 8% to ensure variety.

---

## 🛠 Data Preprocessing and Feature Engineering
Raw text data was processed through a comprehensive pipeline in `processing.py`.

### 1. Domain-Specific Feature Engineering
Before cleaning, specific structural features were extracted based on spam behavior:
* **Caps Ratio:** Ratio of uppercase letters (indicates "shouting").
* **Exclamation Count:** Frequency of exclamation marks.
* **Punctuation Ratio:** Density of punctuation characters.
* **Word Count:** Total number of words.

### 2. Text Cleaning Pipeline
* **Noise Removal:** Removal of HTML tags and URLs via RegEx.
* **Normalization:** Conversion to lowercase.
* **Stopwords Removal:** Standard stopwords removed, but critical keywords (e.g., "you", "win", "free") were explicitly preserved.
* **Lemmatization:** Reducing words to root forms using WordNetLemmatizer.

---

## 🧠 Models and Methodology
Four distinct architectures were benchmarked:

### 1. Logistic Regression (Baseline)
* **Feature Representation:** TF-IDF (Max 5000 features, N-grams).
* **Configuration:** Optimized L2 Regularization via GridSearchCV.
* **Result:** A strong baseline for high-dimensional sparse data.

### 2. Decision Tree Classifier
* **Optimization:** Tuned `max_depth` and `min_samples_split` to balance bias and variance.

### 3. Multi-Layer Perceptron (MLP) - **Best Model** 🏆
Two variations were implemented:
* **MLP + TF-IDF:** Uses the sparse TF-IDF matrix as input.
* **MLP + Word2Vec:** Uses dense vector representations (Skip-gram, sg=1).
* **Architecture:** 2 Hidden Layers (128 and 64 neurons) with ReLU activation.

### 4. Recurrent Neural Network (Bi-LSTM)
* **Goal:** To capture the sequential nature of text.
* **Structure:** Embedding Layer + Bidirectional LSTM (128 units) + Dropout (50%) + Fully Connected Layer.
* **Library:** Implemented in PyTorch.

---

## 🛡 Overfitting Prevention Strategies
To ensure the models generalize well to unseen data, the following techniques were used:
1.  **L2 Regularization:** Used in Logistic Regression to penalize large weights.
2.  **Early Stopping:** Halted training when validation scores ceased to improve (Patience=3 for LSTM).
3.  **Dropout:** Applied in LSTM with a rate of 0.5.
4.  **Cross-Validation:** 5-Fold Cross-Validation used during hyperparameter tuning.

---

## 📈 Results
Models were evaluated on a held-out Test Set (20% of data).

| Model | Accuracy | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (TF-IDF)** | **0.93** | **0.93** | **0.93** | **0.93** |
| Logistic Regression | 0.92 | 0.92 | 0.92 | 0.92 |
| Decision Tree | 0.88 | 0.88 | 0.89 | 0.88 |
| Bi-LSTM (RNN) | 0.84 | 0.84 | 0.84 | 0.84 |
| MLP (Word2Vec) | 0.83 | 0.83 | 0.83 | 0.83 |


### Performance Analysis
* **Best Model:** MLP (TF-IDF) achieved the highest performance (93%). This indicates that for this dataset size, explicit keyword frequency is a stronger predictor than semantic context.
* **LSTM Evaluation:** While the LSTM had lower overall accuracy (84%), it achieved a very high **Recall (0.94)** for the Spam class, making it extremely aggressive in catching spam but prone to false positives.

### Error Analysis
Misclassifications in the baseline model revealed:
* **Short Comments:** Extremely short texts (e.g., "nice", "want update") lacked sufficient lexical evidence.
* **Ambiguity:** Terms like "discord" appeared in both legitimate discussions and spam promotions.

---

## 🏁 Conclusion
This project successfully demonstrated the application of NLP techniques for spam detection. While Deep Learning models like LSTM offer theoretical advantages, this study proves that **Simple Neural Networks (MLP) combined with TF-IDF** provide superior performance when keyword frequency is the dominant feature and data size is limited.


