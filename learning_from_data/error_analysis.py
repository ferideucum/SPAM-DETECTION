import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/processed_text.csv")
X_text = df["processed_text"].values
y = df["label"].values

# Aynı split mantığı
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("results/models/logistic_regression/best_logreg_model.pkl")
vec = joblib.load("results/models/logistic_regression/tfidf_vectorizer.pkl")

X_test = vec.transform(X_test_text)
pred = model.predict(X_test)

# Yanlışları bul
wrong_idx = [i for i,(a,p) in enumerate(zip(y_test, pred)) if a!=p]
print("Wrong count:", len(wrong_idx))


for i in wrong_idx[:15]:
    print("\n---")
    print("True:", y_test[i], "Pred:", pred[i])
    print("Text:", X_test_text[i])


# Hata tiplerini say:
# False Positive: gerçek 0 iken 1 demiş (normal yorumu spam sanmış)
fp = sum((y_test == 0) & (pred == 1))
# False Negative: gerçek 1 iken 0 demiş (spam yorumu normal sanmış)
fn = sum((y_test == 1) & (pred == 0))
print("False Positives (0->1):", fp)
print("False Negatives (1->0):", fn)

# Tahmin olasılıklarını hesapla (spam sınıfı için p(spam))
proba = model.predict_proba(X_test)[:, 1]

# İlk 15 hatalı örnek için p(spam), gerçek etiket, tahmin ve metni yazdır
for i in wrong_idx[:15]:
    print(
        "p(spam)=", round(float(proba[i]), 3),
        "| True:", y_test[i],
        "Pred:", pred[i],
        "| Text:", X_test_text[i]
    )