import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.nn.functional import softmax
import json

TRANSCRIPT_ROOT = "../../RLT/Transcription"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self.labels)


def load_transcripts():
    texts = []
    labels = []
    for label_name in ["Deceptive", "Truthful"]:
        label_dir = os.path.join(TRANSCRIPT_ROOT, label_name)
        label_val = 1 if label_name.lower() == "deceptive" else 0

        for fname in os.listdir(label_dir):
            if fname.endswith(".txt"):
                path = os.path.join(label_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        labels.append(label_val)
    return texts, labels


def train_final_distilbert():
    texts, labels = load_transcripts()
    labels = np.array(labels)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = TranscriptDataset(X_train, y_train.tolist(), tokenizer)
    test_dataset = TranscriptDataset(X_test, y_test.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(4):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred, y_pred_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(classification_report(y_true, y_pred, target_names=["truthful", "deceptive"]))
    print(f"✅ Final Accuracy: {acc:.2f}, F1: {f1:.2f}")

    # Save everything
    os.makedirs("models/distilbert_final", exist_ok=True)
    model.save_pretrained("models/distilbert_final")
    tokenizer.save_pretrained("models/distilbert_final")

    np.save("distilbert_y_true.npy", np.array(y_true))
    np.save("distilbert_y_pred_proba.npy", np.array(y_pred_probs))

    with open("distilbert_results.json", "w") as f:
        json.dump({"model": "DistilBERT", "accuracy": float(acc), "f1": float(f1)}, f)

    print("✅ DistilBERT model, metrics, and ROC data saved!")


if __name__ == "__main__":
    train_final_distilbert()
