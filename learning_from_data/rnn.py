import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import re

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/processed_text.csv"
OUTPUT_DIR = "results/models/rnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hiperparametreler
MAX_LEN = 100  # Her cümlenin bakılacak ilk 100 kelimesi
EMBEDDING_DIM = 100  # Kelime vektörlerinin boyutu
HIDDEN_DIM = 128  # LSTM hafıza hücresi boyutu
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5  # Yönerge şartı: Regularization

# Cihaz ayarı (Mac M1/M2 için 'mps', Nvidia için 'cuda', yoksa 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# =====================================================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["processed_text"])

texts = df["processed_text"].values
labels = df["label"].values

# Train/Val/Test Split (60% Train, 20% Val, 20% Test)
# Önce Train ve Temp (%40) olarak ayır
X_train_raw, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.4, random_state=42, stratify=labels)
# Sonra Temp'i Val ve Test olarak ayır
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")


# =====================================================
# 2. VOCABULARY & TOKENIZATION
# =====================================================
# Kelimeleri sayılara çeviren basit bir yapı
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 2
        for sentence in sentence_list:
            for word in str(sentence).split():
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print(f"Vocab size: {len(self.stoi)}")

    def numericalize(self, text):
        tokenized = [self.stoi.get(word, self.stoi["<UNK>"]) for word in str(text).split()]
        return tokenized


# Vocab sadece Train setiyle oluşturulur (Data Leakage olmaması için)
vocab = Vocabulary(freq_threshold=2)
vocab.build_vocabulary(X_train_raw)


# =====================================================
# 3. DATASET CLASS (PADDING ILE)
# =====================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        numericalized = self.vocab.numericalize(self.texts[index])

        # Padding (Sabit uzunluk)
        if len(numericalized) < self.max_len:
            numericalized += [0] * (self.max_len - len(numericalized))
        else:
            numericalized = numericalized[:self.max_len]

        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


train_dataset = TextDataset(X_train_raw, y_train, vocab, MAX_LEN)
val_dataset = TextDataset(X_val_raw, y_val, vocab, MAX_LEN)
test_dataset = TextDataset(X_test_raw, y_test, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# =====================================================
# 4. LSTM MODEL MIMARISI (DROPOUT DAHIL)
# =====================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # DEĞİŞİKLİK 1: bidirectional=True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        # DEĞİŞİKLİK 2: Bidirectional olduğu için çıktı boyutu 2 katına çıkar (hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)

        # output, (hidden, cell)
        output, (hidden, cell) = self.lstm(embedded)

        # Bidirectional olduğu için son hidden state'leri birleştirmemiz lazım
        # hidden boyutu: [num_layers * num_directions, batch, hidden_dim]
        # Son iki katmanı (ileri ve geri) alıp birleştiriyoruz
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)


model = LSTMClassifier(
    vocab_size=len(vocab.stoi),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=2,  # Binary Classification
    dropout_rate=DROPOUT_RATE
).to(device)

class_weights = torch.tensor([1.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================================
# 5. EĞİTİM DÖNGÜSÜ
# =====================================================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Early Stopping Değişkenleri
best_val_loss = float('inf')
patience = 3   # 3 epoch boyunca iyileşme olmazsa dur
counter = 0

print("Starting training with Early Stopping...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # --- EARLY STOPPING KONTROLÜ ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # En iyi modeli hafızaya al (dosyaya da kaydediyoruz)
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_lstm_model.pth")
        print("  -> Validation loss düştü, model kaydedildi.")
    else:
        counter += 1
        print(f"  -> İyileşme yok. EarlyStopping sayacı: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered! Eğitim durduruluyor.")
            break

# Eğitimin sonunda EN İYİ modeli geri yükle ki test sonuçları (Section 6) en iyi halimizle yapılsın
print("Loading best model for testing...")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_lstm_model.pth"))

# =====================================================
# 6. GÖRSELLEŞTİRME VE TEST
# =====================================================

# A. Learning Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(f"{OUTPUT_DIR}/lstm_learning_curves.png")
print("Learning curves saved.")
#

# B. Test Set Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n========== LSTM RESULTS (TEST SET) ==========")
print(classification_report(all_labels, all_preds))

# C. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title("Confusion Matrix: LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{OUTPUT_DIR}/lstm_confusion_matrix.png")

# Modeli Kaydet
torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_model.pth")
print("Model saved.")