import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from parser import build_dataset

# Load data
bert_embeddings = np.load(
    "../results/text_bert_embeddings.npy", allow_pickle=True
).item()
gesture_data = build_dataset()
audio_mfcc = np.load("../results/audio_mfcc_sequences.npy", allow_pickle=True).item()


class DeceptionDataset(Dataset):
    def __init__(self):
        self.samples = []
        gesture_scaler = StandardScaler()

        gestures = [list(sample["gestures"].values()) for sample in gesture_data]
        gesture_scaler.fit(gestures)

        for sample in gesture_data:
            clip_id = sample["clip_id"]
            if clip_id not in bert_embeddings or clip_id not in audio_mfcc:
                continue

            bert_vec = bert_embeddings[clip_id]
            gest_vec = gesture_scaler.transform([list(sample["gestures"].values())])[0]
            audio_seq = audio_mfcc[clip_id]
            label = 1 if sample["label"] == "deceptive" else 0

            self.samples.append((bert_vec, gest_vec, audio_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bert, gest, audio_seq, label = self.samples[idx]
        return (
            torch.tensor(bert, dtype=torch.float32),
            torch.tensor(gest, dtype=torch.float32),
            torch.tensor(audio_seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_fn(batch):
    bert, gest, audio, labels = zip(*batch)
    lengths = torch.tensor([a.shape[0] for a in audio])
    padded_audio = nn.utils.rnn.pad_sequence(audio, batch_first=True)
    return (
        torch.stack(bert),
        torch.stack(gest),
        padded_audio,
        lengths,
        torch.stack(labels).unsqueeze(1),
    )


class MultiModalLSTM(nn.Module):
    def __init__(self, gesture_dim, bert_dim, audio_dim, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(bert_dim + gesture_dim + hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, bert, gesture, audio, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            audio, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        audio_embed = hn[-1]
        fused = torch.cat((bert, gesture, audio_embed), dim=1)
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


def train_eval_model(hidden_size, dropout, lr):
    dataset = DeceptionDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_data, batch_size=8, collate_fn=collate_fn)

    model = MultiModalLSTM(
        gesture_dim=dataset[0][1].shape[0],
        bert_dim=dataset[0][0].shape[0],
        audio_dim=dataset[0][2].shape[1],
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for bert, gest, audio, lengths, labels in train_loader:
            bert, gest, audio, labels = (
                bert.to(device),
                gest.to(device),
                audio.to(device),
                labels.to(device),
            )
            preds = model(bert, gest, audio, lengths)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"📦 Epoch {epoch+1} - Loss: {total_loss:.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bert, gest, audio, lengths, labels in val_loader:
            preds = model(bert.to(device), gest.to(device), audio.to(device), lengths)
            y_pred.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
            y_true.extend(labels.numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["truthful", "deceptive"]))
    print(
        f"✅ Config: hidden={hidden_size}, dropout={dropout}, lr={lr} → Acc: {acc:.2f}, F1: {f1:.2f}\n"
    )
    return acc, f1


# Run hyperparameter search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = [(64, 0.3, 1e-4), (128, 0.3, 1e-4), (128, 0.5, 1e-4), (64, 0.5, 3e-4)]

for h, d, lr in configs:
    print(f"\n🚀 Running config: hidden={h}, dropout={d}, lr={lr}\n")
    train_eval_model(h, d, lr)
