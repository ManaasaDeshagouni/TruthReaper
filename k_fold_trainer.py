import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from sequence_lstm_trainer import (
    TruthReaperLSTM,
    pad_or_truncate,
    min_max_normalize,
    normalize_summary_features,
    MAX_LEN
)

SUMMARY_DIM = 12
FEATURE_DIM = 4
EPOCHS = 75
BATCH_SIZE = 8
K = 5  # Number of folds

def load_dataset(path="sequence_dataset_combined.json"):
    with open(path, "r") as f:
        return json.load(f)

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.X_seq, self.X_summary, self.y = [], [], []

        for entry in data:
            pitch = min_max_normalize(pad_or_truncate(entry["pitch_seq"], MAX_LEN))
            energy = min_max_normalize(pad_or_truncate(entry["energy_seq"], MAX_LEN))
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

def train_and_evaluate(train_idx, test_idx, full_dataset):
    model = TruthReaperLSTM(input_dim=FEATURE_DIM, hidden_dim=128, summary_dim=SUMMARY_DIM, output_dim=2)
    train_set = Subset(full_dataset, train_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    class_weights = torch.tensor([1.0, 1.2])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for (x_seq, x_summary), y in train_loader:
            optimizer.zero_grad()
            logits = model(x_seq, x_summary)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x_seq, x_summary), y in test_loader:
            logits = model(x_seq, x_summary)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "detailed": classification_report(all_labels, all_preds, target_names=["truth", "lie"])
    }

def plot_metrics(all_metrics):
    folds = list(range(1, len(all_metrics["accuracy"]) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(folds, all_metrics["accuracy"], label="Accuracy", marker="o")
    plt.plot(folds, all_metrics["precision"], label="Precision", marker="o")
    plt.plot(folds, all_metrics["recall"], label="Recall", marker="o")
    plt.plot(folds, all_metrics["f1"], label="F1 Score", marker="o")
    plt.xticks(folds)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("K-Fold Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kfold_metrics.png")
    print("📸 Saved plot to kfold_metrics.png")
    plt.show()

def run_k_fold():
    print("📚 Loading dataset...")
    raw_data = load_dataset()
    dataset = SequenceDataset(raw_data)
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    all_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n🔁 Fold {fold+1}/{K}")
        metrics = train_and_evaluate(train_idx, test_idx, dataset)
        print(metrics["detailed"])
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    print("\n📊 Average Results Across Folds:")
    for key in all_metrics:
        avg = np.mean(all_metrics[key])
        print(f"{key.capitalize()}: {avg:.4f}")

    plot_metrics(all_metrics)

if __name__ == "__main__":
    run_k_fold()