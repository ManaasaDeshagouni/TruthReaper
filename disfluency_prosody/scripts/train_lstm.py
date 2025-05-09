import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load all features
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc_seq = np.load("../results/audio_mfcc_sequences.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

X_seq, X_bert, X_extra, y = [], [], [], []

for clip_id in bert:
    if clip_id in mfcc_seq and clip_id in disf and clip_id in pros:
        try:
            mfcc = mfcc_seq[clip_id]
            bert_vec = bert[clip_id]
            disf_vec = disf[clip_id]
            pros_vec = pros[clip_id]

            if mfcc.ndim == 2 and mfcc.shape[1] == 13:
                padded = (
                    mfcc[:50]
                    if mfcc.shape[0] >= 50
                    else np.pad(mfcc, ((0, 50 - mfcc.shape[0]), (0, 0)))
                )
                X_seq.append(padded)
                X_bert.append(bert_vec)
                X_extra.append(np.concatenate([disf_vec, pros_vec]))
                y.append(1 if "lie" in clip_id else 0)
        except Exception as e:
            print(f"⚠️ Skipping {clip_id}: {e}")

X_seq = np.array(X_seq)
X_bert = np.array(X_bert)
X_extra = np.array(X_extra)
y = np.array(y)


class FusionLSTM(nn.Module):
    def __init__(self, audio_dim=13, lstm_hidden=32, bert_dim=768, extra_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + bert_dim + extra_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio_seq, bert_vec, extra_feats):
        _, (hn, _) = self.lstm(audio_seq)
        x = torch.cat([hn[-1], bert_vec, extra_feats], dim=1)
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_seq, y)):
    print(f"\n📂 Fold {fold+1}")

    X_train_seq = torch.tensor(X_seq[train_idx], dtype=torch.float32).to(device)
    X_train_bert = torch.tensor(X_bert[train_idx], dtype=torch.float32).to(device)
    X_train_extra = torch.tensor(X_extra[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1).to(device)

    X_test_seq = torch.tensor(X_seq[test_idx], dtype=torch.float32).to(device)
    X_test_bert = torch.tensor(X_bert[test_idx], dtype=torch.float32).to(device)
    X_test_extra = torch.tensor(X_extra[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_train_seq, X_train_bert, X_train_extra, y_train)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = FusionLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for seqs, berts, extras, labels in train_loader:
            preds = model(seqs, berts, extras)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_seq, X_test_bert, X_test_extra)
        final = (preds > 0.5).float()
        acc = accuracy_score(y_test.cpu().numpy(), final.cpu().numpy())
        f1 = f1_score(y_test.cpu().numpy(), final.cpu().numpy())
        acc_list.append(acc)
        f1_list.append(f1)

        print(
            classification_report(
                y_test.cpu().numpy(),
                final.cpu().numpy(),
                target_names=["truthful", "deceptive"],
            )
        )
        print(f"✅ Accuracy: {acc:.2f}, F1: {f1:.2f}")

print("\n🔚 Final LSTM Fusion Avg")
print(f"✅ Accuracy: {np.mean(acc_list):.2f}, F1: {np.mean(f1_list):.2f}")
import json

with open("lstm_fusion_results.json", "w") as f:
    json.dump(
        {
            "model": "LSTM (Fusion)",
            "accuracy": np.mean(acc_list),
            "f1": np.mean(f1_list),
        },
        f,
    )
