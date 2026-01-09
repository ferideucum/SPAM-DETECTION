import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# =========================
# CONFIG
# =========================
DATA_PATH = "data/raw/dataset.csv"
OUTPUT_DIR = "results/eda"
TEXT_COL = "comment"
LABEL_COL = "label"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# =========================
# BASIC INFO
# =========================
print("\nMissing values:")
print(df.isnull().sum())

# =========================
# LABEL DISTRIBUTION
# =========================
label_counts = df[LABEL_COL].value_counts()

plt.figure()
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title("Class Distribution (0 = Ham, 1 = Spam)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
plt.close()

print("\nLabel distribution:")
print(label_counts)

# =========================
# TEXT LENGTH ANALYSIS
# =========================
df["char_length"] = df[TEXT_COL].apply(len)
df["word_length"] = df[TEXT_COL].apply(lambda x: len(x.split()))

plt.figure()
sns.histplot(df["word_length"], bins=50)
plt.title("Word Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(f"{OUTPUT_DIR}/word_length_distribution.png")
plt.close()

plt.figure()
sns.boxplot(x=df[LABEL_COL], y=df["word_length"])
plt.title("Word Length by Class")
plt.xlabel("Label (0 = Ham, 1 = Spam)")
plt.ylabel("Word Count")
plt.savefig(f"{OUTPUT_DIR}/word_length_by_class.png")
plt.close()

print("\nWord length statistics:")
print(df.groupby(LABEL_COL)["word_length"].describe())

# =========================
# MOST COMMON WORDS (RAW)
# =========================
def get_top_words(text_series, n=20):
    words = []
    for text in text_series:
        tokens = re.findall(r"\b\w+\b", text.lower())
        words.extend(tokens)
    return Counter(words).most_common(n)

spam_words = get_top_words(df[df[LABEL_COL] == 1][TEXT_COL])
ham_words = get_top_words(df[df[LABEL_COL] == 0][TEXT_COL])

print("\nTop words in SPAM comments:")
for w, c in spam_words:
    print(f"{w}: {c}")

print("\nTop words in HAM comments:")
for w, c in ham_words:
    print(f"{w}: {c}")

# =========================
# SAMPLE COMMENTS
# =========================
print("\nSample HAM comments:")
print(df[df[LABEL_COL] == 0][TEXT_COL].head(5).to_string(index=False))

print("\nSample SPAM comments:")
print(df[df[LABEL_COL] == 1][TEXT_COL].head(5).to_string(index=False))

# =========================
# SAVE SUMMARY
# =========================
with open(f"{OUTPUT_DIR}/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("EDA SUMMARY\n")
    f.write("====================\n\n")
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write("Label distribution:\n")
    f.write(label_counts.to_string())
    f.write("\n\nWord length stats by class:\n")
    f.write(df.groupby(LABEL_COL)["word_length"].describe().to_string())