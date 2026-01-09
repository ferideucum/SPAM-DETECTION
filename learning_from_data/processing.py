import pandas as pd
import re
import nltk
import string
import numpy as np
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# =========================
# SSL SERTİFİKA HATASI ÇÖZÜMÜ (MAC İÇİN)
# =========================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# =========================
# CONFIG
# =========================
# ... Kodun geri kalanı aynen devam etsin ...

# =========================
# CONFIG
# =========================
INPUT_PATH = "data/raw/dataset.csv"
OUTPUT_PATH = "data/processed/processed_text.csv"

TEXT_COL = "comment"
LABEL_COL = "label"

os.makedirs("data/processed", exist_ok=True)

# =========================
# NLTK RESOURCES
# =========================
# İlk çalıştırmada indirmediysen hata vermesin diye check
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# =========================
# STOP WORDS & LEMMATIZER
# =========================
stop_words = set(stopwords.words("english"))
# Bazı kelimeler duygu analizi/spam için önemli olabilir, onları silmeyelim
domain_keep_words = {"you", "your", "free", "win", "won", "not", "no", "offer"}
stop_words = stop_words - domain_keep_words

lemmatizer = WordNetLemmatizer()


# =========================
# 1. CUSTOM FEATURE ENGINEERING
# =========================
# Bu fonksiyon metin TEMİZLENMEDEN ÖNCE çalışmalı.
# Çünkü temizlikte noktalama işaretlerini ve büyük harfleri kaybediyoruz.

def extract_custom_features(text):
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0])

    # A. Kelime Sayısı
    word_count = len(text.split())

    # B. Karakter Sayısı
    char_count = len(text)

    # C. Büyük Harf Oranı (Caps Lock bağırma/spam belirtisidir)
    # Sıfıra bölünme hatasını önlemek için (char_count + 1)
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / (char_count + 1)

    # D. Ünlem İşareti Sayısı (Duygu yoğunluğu)
    exclamation_count = text.count("!")

    # E. Noktalama İşareti Oranı
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_count / (char_count + 1)

    return pd.Series([word_count, caps_ratio, exclamation_count, punct_ratio])


# =========================
# 2. TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 1. HTML tagleri veya URL'leri temizle (Basit regex)
    text = re.sub(r'http\S+', '', text)

    # 2. Sadece harfleri tut, gerisini sil, küçük harfe çevir
    # (Burada noktalama işaretleri uçuyor)
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()

    # 3. Tokenization
    tokens = text.split()

    # 4. Stop word removal & Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

    # 5. Tekrar birleştir
    return " ".join(tokens)


# =========================
# MAIN PIPELINE
# =========================
print("Loading data...")
df = pd.read_csv(INPUT_PATH)

# Boş satırları uçur
df = df.dropna(subset=[TEXT_COL])

print("Extracting Custom Features (Domain Knowledge)...")
# Yeni sütunları ekliyoruz
feature_cols = ["word_count", "caps_ratio", "exclamation_count", "punct_ratio"]
df[feature_cols] = df[TEXT_COL].apply(extract_custom_features)

print("Cleaning Text...")
df["processed_text"] = df[TEXT_COL].apply(preprocess_text)

# Boş kalan textleri (sadece stopword olan cümleler boş kalabilir) temizle
df = df[df["processed_text"].str.len() > 0]

# =========================
# SAVE
# =========================
# Artık elimizde hem temiz metin hem de sayısal özellikler var.
output_columns = [LABEL_COL, "processed_text"] + feature_cols
df[output_columns].to_csv(OUTPUT_PATH, index=False)

print("Preprocessing completed.")
print(f"Data saved to: {OUTPUT_PATH}")
print("\nSample processed data with custom features:")
print(df.head())