import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MAX_LEN = 300
FEATURE_DIM = 4   # pitch, energy, hesitation, disfluency
SUMMARY_DIM = 12  # Number of additional summary features

# Min-Max Normalization Function for sequences
def min_max_normalize(seq):
    arr = np.array(seq)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return arr
    return (arr - min_val) / (max_val - min_val)

# Pad or Truncate Function
def pad_or_truncate(seq, target_len=300):
    seq = seq[:target_len]
    if len(seq) < target_len:
        seq = seq + [0.0] * (target_len - len(seq))
    return seq

# Normalize Summary Features (Min-Max Scaling)
def normalize_summary_features(summary_list):
    arr = np.array(summary_list)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    scaled = (arr - min_vals) / (max_vals - min_vals + 1e-8)
    return torch.tensor(scaled, dtype=torch.float32)

# Custom Dataset
class SequenceDataset(Dataset):
    def __init__(self, data):
        self.X_seq = []
        self.X_summary = []
        self.y = []

        for entry in data:
            pitch = 0.7*min_max_normalize(pad_or_truncate(entry["pitch_seq"], MAX_LEN))
            energy = 0.7*min_max_normalize(pad_or_truncate(entry["energy_seq"], MAX_LEN))
            hes = pad_or_truncate(entry["hesitation_seq"], MAX_LEN)
            disf = pad_or_truncate(entry["disfluency_seq"], MAX_LEN)

            seq_features = np.stack([pitch, energy, hes, disf], axis=1)
            self.X_seq.append(seq_features)

            summary_feats = [
                entry["total_pause_time"],
                entry["avg_pause_duration"],
                entry["long_pause_count"],
                entry["total_disfluencies"],
                entry["filler_count"],
                entry["stutter_count"],
                entry["phrase_repetition"],
                entry["disfluency_rate"],
                entry["pitch_variance"],
                entry["avg_energy"],
                entry["energy_variance"],
                entry["prosodic_activity"]
            ]
            self.X_summary.append(summary_feats)
            self.y.append(1 if entry["label"] == "lie" else 0)

        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.X_summary = normalize_summary_features(self.X_summary)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (self.X_seq[idx], self.X_summary[idx]), self.y[idx]

# Hybrid LSTM Model
class TruthReaperLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, summary_dim=SUMMARY_DIM, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2 + summary_dim, output_dim)

    def forward(self, x_seq, x_summary):
        _, (hn, _) = self.lstm(x_seq)
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        combined = torch.cat((hn, x_summary), dim=1)
        out = self.dropout(combined)
        out = self.fc(out)
        return out

# Load Dataset
def load_dataset(path="sequence_dataset_combined.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data

# Training Function
def train_model():
    print("📚 Loading dataset...")
    data = load_dataset()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_set = SequenceDataset(train_data)
    test_set = SequenceDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)

    model = TruthReaperLSTM()
    class_weights = torch.tensor([1.0, 1.05])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("🚀 Training Hybrid LSTM model...")
    for epoch in range(75):
        model.train()
        total_loss = 0
        for (X_seq_batch, X_summary_batch), y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_seq_batch, X_summary_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (X_seq_batch, X_summary_batch), y_batch in test_loader:
            logits = model(X_seq_batch, X_summary_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["truth", "lie"]))

    torch.save(model.state_dict(), "truthreaper_hybrid_lstm.pt")
    print("✅ Model saved as truthreaper_hybrid_lstm.pt")

if __name__ == "__main__":
    train_model()