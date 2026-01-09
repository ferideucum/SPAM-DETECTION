import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/processed_text.csv"
OUTPUT_DIR = "results/models/logistic_regression"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. VERİYİ YÜKLE VE HAZIRLA
# =====================================================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["processed_text"])

X_text = df["processed_text"].values
y = df["label"].values

print("Vectorizing data (TF-IDF)...")
# Logistic Regression yüksek boyutlu veride (TF-IDF) genelde iyi çalışır.
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 2. HYPERPARAMETER TUNING (L1/L2 REGULARIZATION)
# =====================================================
print("Starting GridSearchCV...")

# Yönergedeki "Regularization & Overfitting Prevention" maddesi için:
# C küçüldükçe regularization artar (Overfitting azalır).
# Penalty L1 veya L2 denenerek en iyisi seçilir.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # liblinear hem l1 hem l2 destekler
}

log_reg = LogisticRegression(random_state=42, max_iter=1000)

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# =========================
# 3. MODEL DEĞERLENDİRME
# =====================================================
y_pred = best_model.predict(X_test)
# ROC Curve için olasılık değerleri lazım
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n========== TEST SET RESULTS ==========")
print(classification_report(y_test, y_pred))

# --- A. Confusion Matrix ---
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# --- B. ROC Curve & AUC (Binary Class için Zorunlu) ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()
print(f"ROC Curve saved (AUC: {roc_auc:.2f})")

# --- C. Learning Curve ---
print("Generating Learning Curve...")
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker='o')
plt.title("Learning Curve: Logistic Regression")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/learning_curve.png")
plt.close()

# =========================
# 4. KAYDET
# =====================================================
joblib.dump(best_model, f"{OUTPUT_DIR}/best_logreg_model.pkl")
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")
print("All results saved.")