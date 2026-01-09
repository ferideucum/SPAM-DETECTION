import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # Modeli kaydetmek için

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/processed_text.csv"
OUTPUT_DIR = "results/models/decision_tree"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. VERİYİ YÜKLE
# =====================================================
df = pd.read_csv(DATA_PATH)
# NaN kontrolü (Veri setinde boş satır varsa patlamasın)
df = df.dropna(subset=["processed_text"])

X_text = df["processed_text"].values
y = df["label"].values

# =========================
# 2. FEATURE ENGINEERING (TF-IDF)
# =====================================================
# Decision Tree genelde TF-IDF ile BoW'dan daha iyi çalışır,
# bu yüzden sadece TF-IDF üzerinden giderek kodu sadeleştirdim.
print("Vectorizing data (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(X_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. HYPERPARAMETER TUNING (GridSearch)
# =====================================================
print("Starting GridSearchCV for Hyperparameter Tuning...")

# Denenecek parametreler
param_grid = {
    'max_depth': [10, 20, 50, None],
    'min_samples_split': [2, 10, 20],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)

# cv=5 -> Yönergedeki "At least 5-fold cross-validation" şartını sağlar
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"\nBest Parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# =========================
# 4. MODEL DEĞERLENDİRME
# =====================================================
y_pred = best_model.predict(X_test)

print("\n========== TEST SET RESULTS ==========")
print(classification_report(y_test, y_pred))

# --- A. Confusion Matrix Görselleştirme ---
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()
print(f"Confusion Matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# --- B. Learning Curve (Bias-Variance Analizi İçin) ---
# Bu grafik raporundaki "Bias-Variance analysis" başlığı için kritik.
print("\nGenerating Learning Curve (This may take a moment)...")

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker='o')
plt.title("Learning Curve: Decision Tree")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/learning_curve.png")
plt.close()
print(f"Learning Curve saved to {OUTPUT_DIR}/learning_curve.png")

# =========================
# 5. MODELİ KAYDET
# =====================================================
joblib.dump(best_model, f"{OUTPUT_DIR}/best_dt_model.pkl")
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")
print("Model saved successfully.")