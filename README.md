# SPAM-DETECTION
# SPAM DETECTION AND TEXT CLASSIFICATION PROJECT

[cite_start]This project was developed by students of Fatih Sultan Mehmet Vakıf University to perform binary text classification (Spam vs. Ham) using Machine Learning and Deep Learning techniques[cite: 1, 2, 3].


## 👥 Project Team
* [cite_start]**Feride Uçum** - 2221251022 [cite: 5]
* [cite_start]**Furkan Emre Karçıkar** - 2221251007 [cite: 6]
* [cite_start]**Semih Alev** - 2221251010 [cite: 7]

---

## 📝 Abstract
[cite_start]In this project, a robust system for binary text classification (Spam vs. Ham) was developed and analyzed[cite: 9]. [cite_start]The study explores the transition from classical Machine Learning algorithms (Logistic Regression, Decision Trees) to modern Deep Learning architectures (Multi-Layer Perceptrons, Bi-LSTM)[cite: 10].

[cite_start]A significant portion of the work focused on **Preprocessing**, **Feature Engineering** (extracting domain-specific features), and **Overfitting Prevention** mechanisms such as Early Stopping, Dropout, and L2 Regularization[cite: 11].

[cite_start]The results demonstrate that the **MLP model utilizing TF-IDF** achieved the highest accuracy (**93%**), outperforming complex sequential models for this specific dataset[cite: 12].

---

## 📊 Data Collection
[cite_start]The dataset was collected from YouTube video comments using Python-based web scraping scripts and the `youtube_comment_downloader` library[cite: 24, 25].

* [cite_start]**Total Data:** 2339 Comments [cite: 31]
* [cite_start]**Normal (Ham) Comments:** 1170 (Label: 0) [cite: 32]
* [cite_start]**Spam Comments:** 1169 (Label: 1) [cite: 32]
* [cite_start]**Source:** 15 Normal video URLs, 42 Spam video URLs[cite: 28].

**Collection Strategy:**
* [cite_start]**Normal Comments:** Filtered for English only, checked against spam-like signals (links, contact info), and deduplicated[cite: 42, 45, 47].
* [cite_start]**Spam Comments:** Targeted specific spam patterns (URLs, CTA), with "URL-only" spam capped at 8% to ensure variety[cite: 53, 57].

---

## 🛠 Data Preprocessing and Feature Engineering
[cite_start]Raw text data was processed through a comprehensive pipeline in `processing.py`[cite: 70].

### 1. Domain-Specific Feature Engineering
[cite_start]Before cleaning, specific structural features were extracted based on spam behavior[cite: 72, 73]:
* [cite_start]**Caps Ratio:** Ratio of uppercase letters (indicates "shouting")[cite: 74, 75].
* [cite_start]**Exclamation Count:** Frequency of exclamation marks[cite: 76].
* [cite_start]**Punctuation Ratio:** Density of punctuation characters[cite: 77].
* [cite_start]**Word Count:** Total number of words[cite: 78].

### 2. Text Cleaning Pipeline
* [cite_start]**Noise Removal:** Removal of HTML tags and URLs via RegEx[cite: 81].
* [cite_start]**Normalization:** Conversion to lowercase[cite: 82].
* [cite_start]**Stopwords Removal:** Standard stopwords removed, but critical keywords (e.g., "you", "win", "free") were explicitly preserved[cite: 83].
* [cite_start]**Lemmatization:** Reducing words to root forms using WordNetLemmatizer[cite: 84].

---

## 🧠 Models and Methodology
[cite_start]Four distinct architectures were benchmarked[cite: 20]:

### 1. Logistic Regression (Baseline)
* [cite_start]**Feature Representation:** TF-IDF (Max 5000 features, N-grams)[cite: 137, 138].
* [cite_start]**Configuration:** Optimized L2 Regularization via GridSearchCV[cite: 139].
* [cite_start]**Result:** A strong baseline for high-dimensional sparse data[cite: 140].

### 2. Decision Tree Classifier
* [cite_start]**Optimization:** Tuned `max_depth` and `min_samples_split` to balance bias and variance[cite: 175, 176].

### 3. Multi-Layer Perceptron (MLP) - **Best Model** 🏆
Two variations were implemented:
* [cite_start]**MLP + TF-IDF:** Uses the sparse TF-IDF matrix as input[cite: 204].
* [cite_start]**MLP + Word2Vec:** Uses dense vector representations (Skip-gram, sg=1)[cite: 225].
* [cite_start]**Architecture:** 2 Hidden Layers (128 and 64 neurons) with ReLU activation[cite: 251].

### 4. Recurrent Neural Network (Bi-LSTM)
* [cite_start]**Goal:** To capture the sequential nature of text[cite: 253].
* [cite_start]**Structure:** Embedding Layer + Bidirectional LSTM (128 units) + Dropout (50%) + Fully Connected Layer[cite: 256, 307].
* [cite_start]**Library:** Implemented in PyTorch[cite: 253].

---

## 🛡 Overfitting Prevention Strategies
[cite_start]To ensure the models generalize well to unseen data, the following techniques were used[cite: 298]:
1.  [cite_start]**L2 Regularization:** Used in Logistic Regression to penalize large weights[cite: 301].
2.  [cite_start]**Early Stopping:** Halted training when validation scores ceased to improve (Patience=3 for LSTM)[cite: 303, 304].
3.  [cite_start]**Dropout:** Applied in LSTM with a rate of 0.5[cite: 307].
4.  [cite_start]**Cross-Validation:** 5-Fold Cross-Validation used during hyperparameter tuning[cite: 310].

---

## 📈 Results
[cite_start]Models were evaluated on a held-out Test Set (20% of data)[cite: 312].

| Model | Accuracy | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (TF-IDF)** | **0.93** | **0.93** | **0.93** | **0.93** |
| Logistic Regression | 0.92 | 0.92 | 0.92 | 0.92 |
| Decision Tree | 0.88 | 0.88 | 0.89 | 0.88 |
| Bi-LSTM (RNN) | 0.84 | 0.84 | 0.84 | 0.84 |
| MLP (Word2Vec) | 0.83 | 0.83 | 0.83 | 0.83 |
[cite_start][cite: 314]

### Performance Analysis
* **Best Model:** MLP (TF-IDF) achieved the highest performance (93%). [cite_start]This indicates that for this dataset size, explicit keyword frequency is a stronger predictor than semantic context[cite: 316, 317].
* [cite_start]**LSTM Evaluation:** While the LSTM had lower overall accuracy (84%), it achieved a very high **Recall (0.94)** for the Spam class, making it extremely aggressive in catching spam but prone to false positives[cite: 320, 321].

### Error Analysis
Misclassifications in the baseline model revealed:
* [cite_start]**Short Comments:** Extremely short texts (e.g., "nice", "want update") lacked sufficient lexical evidence[cite: 374].
* [cite_start]**Ambiguity:** Terms like "discord" appeared in both legitimate discussions and spam promotions[cite: 380].

---

## 🏁 Conclusion
[cite_start]This project successfully demonstrated the application of NLP techniques for spam detection[cite: 383]. [cite_start]While Deep Learning models like LSTM offer theoretical advantages, this study proves that **Simple Neural Networks (MLP) combined with TF-IDF** provide superior performance when keyword frequency is the dominant feature and data size is limited[cite: 384].

---
[cite_start]*This project was prepared within the scope of the Computer Engineering Department at Fatih Sultan Mehmet Vakıf University.* [cite: 1]
