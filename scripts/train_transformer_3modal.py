import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from parser import build_dataset

bert_embeddings = np.load(
    "../results/text_bert_embeddings.npy", allow_pickle=True
).item()
gesture_data = build_dataset()
audio_mfcc = np.load("../results/audio_mfcc_sequences.npy", allow_pickle=True).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class AudioTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=64, nhead=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, T, 13]
        x = self.input_proj(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class TransformerFusionModel(nn.Module):
    def __init__(self, gesture_dim, bert_dim, audio_dim, audio_model_dim=64):
        super().__init__()
        self.audio_encoder = AudioTransformer(
            input_dim=audio_dim, d_model=audio_model_dim
        )
        self.fc1 = nn.Linear(bert_dim + gesture_dim + audio_model_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, bert, gesture, audio):
        audio_embed = self.audio_encoder(audio)
        x = torch.cat((bert, gesture, audio_embed), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


def k_fold_transformer_training():
    dataset = DeceptionDataset()
    y_all = [int(s[3].item()) for s in dataset]
    skf = StratifiedKFold(
        n_splits=2, shuffle=True, random_state=42
    )  # 🔁 Just 2 folds for speed

    acc_list, f1_list = [], []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(dataset)), y_all)
    ):
        print(f"\n🔁 Fold {fold+1} of 2")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_set, batch_size=8, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_set, batch_size=8, collate_fn=collate_fn)

        model = TransformerFusionModel(
            gesture_dim=dataset[0][1].shape[0],
            bert_dim=dataset[0][0].shape[0],
            audio_dim=dataset[0][2].shape[1],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.BCELoss()

        best_val_loss = float("inf")
        patience, patience_counter = 3, 0

        for epoch in range(15):
            model.train()
            train_loss = 0
            for bert, gest, audio, lengths, labels in train_loader:
                bert, gest, audio, labels = (
                    bert.to(device),
                    gest.to(device),
                    audio.to(device),
                    labels.to(device),
                )
                preds = model(bert, gest, audio)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            # Eval
            model.eval()
            val_loss = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for bert, gest, audio, lengths, labels in val_loader:
                    preds = model(bert.to(device), gest.to(device), audio.to(device))
                    loss = loss_fn(preds.cpu(), labels)
                    val_loss += loss.item()
                    y_pred.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
                    y_true.extend(labels.numpy().flatten())

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print(
                f"📦 Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.2f} | F1: {f1:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹ Early stopping triggered.")
                    break

        print(f"✅ Fold {fold+1} Result — Accuracy = {acc:.2f}, F1 = {f1:.2f}")
        acc_list.append(acc)
        f1_list.append(f1)

    print("\n📊 Final Transformer (2-Fold) Results:")
    print(f"✅ AVG Accuracy = {np.mean(acc_list):.2f}, AVG F1 = {np.mean(f1_list):.2f}")


if __name__ == "__main__":
    k_fold_transformer_training()
