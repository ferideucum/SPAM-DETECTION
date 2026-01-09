import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import Word2Vec

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/processed_text.csv"
OUTPUT_DIR = "results/models/mlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 1. VERİYİ AL VE TEMİZLE
# =====================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["processed_text"]) # Boş satırları at

texts = df["processed_text"].values
y = df["label"].values

# Train/Test Split (Her iki yöntem için ortak)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# FONKSİYON: SONUÇLARI GÖRSELLEŞTİR
# =====================================================
def plot_results(model, model_name, y_test, y_pred):
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_confusion_matrix.png")
    plt.close()

    # 2. Learning Curve (Loss Curve)
    # Sklearn MLP, eğitim sırasındaki loss değerlerini 'loss_curve_' içinde tutar
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(8, 6))
        plt.plot(model.loss_curve_, label="Training Loss")
        if hasattr(model, 'validation_scores_') and model.validation_scores_:
             plt.plot(model.validation_scores_, label="Validation Score")
        plt.title(f"Learning Curve: {model_name}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/{model_name}_loss_curve.png")
        plt.close()

# =====================================================
# YÖNTEM A: TF-IDF + MLP
# =====================================================
print("\n--- Training MLP with TF-IDF ---")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# MLP Yapılandırması:
# hidden_layer_sizes=(128, 64) -> İki gizli katman
# early_stopping=True -> Regularization (Overfitting önler)
# validation_fraction=0.1 -> Eğitimin %10'unu doğrulama için ayırır
mlp_tfidf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    batch_size=64,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True  # Eğitimi canlı izlemek için
)

mlp_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = mlp_tfidf.predict(X_test_tfidf)

print("\n========== TF-IDF + MLP RESULTS ==========")
print(classification_report(y_test, y_pred_tfidf))
plot_results(mlp_tfidf, "MLP_TFIDF", y_test, y_pred_tfidf)

# =====================================================
# YÖNTEM B: WORD2VEC + MLP (Embedding Approach)
# =====================================================
print("\n--- Training MLP with Word2Vec ---")

# 1. Tokenization
tokenized_train = [str(text).split() for text in X_train_text]
tokenized_test = [str(text).split() for text in X_test_text]

# 2. Word2Vec Modelini Eğit (Sadece Train seti üzerinde!)
w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,  # Her kelime 100 boyutlu vektör olacak
    window=5,
    min_count=2,
    workers=4,
    sg=1  # Skip-gram
)

# 3. Cümleleri Vektöre Dönüştür (Average Word Vectors)
def get_sentence_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X_train_w2v = np.array([get_sentence_vector(s, w2v_model) for s in tokenized_train])
X_test_w2v = np.array([get_sentence_vector(s, w2v_model) for s in tokenized_test])

# 4. MLP Eğitimi (Word2Vec Features)
mlp_w2v = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

mlp_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = mlp_w2v.predict(X_test_w2v)

print("\n========== WORD2VEC + MLP RESULTS ==========")
print(classification_report(y_test, y_pred_w2v))
plot_results(mlp_w2v, "MLP_Word2Vec", y_test, y_pred_w2v)

# =====================================================
# KAYDET
# =====================================================
joblib.dump(mlp_tfidf, f"{OUTPUT_DIR}/best_mlp_tfidf.pkl")
print(f"\nResults and plots saved to: {OUTPUT_DIR}")