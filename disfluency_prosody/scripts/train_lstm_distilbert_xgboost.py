import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load features
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc_seq = np.load("../results/audio_mfcc_sequences.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

# Build dataset
X_seq, X_vec, y = [], [], []
for clip_id in bert:
    if clip_id in mfcc_seq and clip_id in disf and clip_id in pros:
        mfcc = mfcc_seq[clip_id]
        if mfcc.shape[0] >= 50:
            padded = mfcc[:50]
        else:
            padded = np.pad(mfcc, ((0, 50 - mfcc.shape[0]), (0, 0)))

        X_seq.append(padded)  # (50,13)
        X_vec.append(
            np.concatenate([bert[clip_id], disf[clip_id], pros[clip_id]])
        )  # (768+3+5)
        y.append(1 if "lie" in clip_id else 0)

X_seq = np.array(X_seq)  # (N, 50, 13)
X_vec = np.array(X_vec)  # (N, 776)
y = np.array(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --------------------------------------------
# LSTM Fusion Model
class FusionLSTM(nn.Module):
    def __init__(self, audio_dim=13, hidden_dim=32, vec_dim=776):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + vec_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq, vec):
        _, (hn, _) = self.lstm(seq)  # seq -> (batch, 50, 13)
        x = torch.cat([hn[-1], vec], dim=1)
        return self.fc(x)


# --------------------------------------------
# 1. Train LSTM (Fusion)
print("✅ Training LSTM Fusion...")

acc_list, f1_list = [], []
all_y_true, all_y_pred_proba = [], []

for train_idx, test_idx in skf.split(X_vec, y):
    X_train_seq = torch.tensor(X_seq[train_idx], dtype=torch.float32).to(device)
    X_train_vec = torch.tensor(X_vec[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1).to(device)

    X_test_seq = torch.tensor(X_seq[test_idx], dtype=torch.float32).to(device)
    X_test_vec = torch.tensor(X_vec[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1).to(device)

    model = FusionLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(10):
        preds = model(X_train_seq, X_train_vec)
        loss = loss_fn(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_seq, X_test_vec)
        all_y_true.extend(y[test_idx].tolist())
        all_y_pred_proba.extend(preds.squeeze().cpu().numpy().tolist())

np.save("lstm_fusion_y_true.npy", np.array(all_y_true))
np.save("lstm_fusion_y_pred_proba.npy", np.array(all_y_pred_proba))

with open("lstm_fusion_results.json", "w") as f:
    json.dump(
        {
            "model": "LSTM (Fusion)",
            "accuracy": float(
                accuracy_score(all_y_true, np.array(all_y_pred_proba) > 0.5)
            ),
            "f1": float(f1_score(all_y_true, np.array(all_y_pred_proba) > 0.5)),
        },
        f,
    )

torch.save(model.state_dict(), "lstm_fusion_model.pt")
print("✅ LSTM Fusion Training Complete.")

# --------------------------------------------
# 2. Train XGBoost

print("✅ Training XGBoost...")

acc_list, f1_list = [], []
all_y_true, all_y_pred_proba = [], []

for train_idx, test_idx in skf.split(X_vec, y):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_vec[train_idx], y[train_idx])
    preds = model.predict(X_vec[test_idx])
    probas = model.predict_proba(X_vec[test_idx])[:, 1]

    acc_list.append(accuracy_score(y[test_idx], preds))
    f1_list.append(f1_score(y[test_idx], preds))

    all_y_true.extend(y[test_idx].tolist())
    all_y_pred_proba.extend(probas.tolist())

np.save("xgboost_y_true.npy", np.array(all_y_true))
np.save("xgboost_y_pred_proba.npy", np.array(all_y_pred_proba))

with open("xgboost_results.json", "w") as f:
    json.dump(
        {
            "model": "XGBoost",
            "accuracy": float(np.mean(acc_list)),
            "f1": float(np.mean(f1_list)),
        },
        f,
    )

joblib.dump(model, "xgboost_model.pkl")
print("✅ XGBoost Training Complete.")

# --------------------------------------------
# 3. Train DistilBERT

print("✅ Training DistilBERT...")

texts = []
labels = []
for clip_id in bert:
    if clip_id in mfcc_seq and clip_id in disf and clip_id in pros:
        labels.append(1 if "lie" in clip_id else 0)
        texts.append(
            "This is a dummy placeholder text."
        )  # Dummy because no real transcript

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
acc_list, f1_list = [], []
all_y_true, all_y_pred_proba = [], []

for train_idx, test_idx in skf.split(texts, labels):
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, return_tensors="pt"
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, return_tensors="pt"
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(4):
        outputs = model(
            input_ids=train_encodings["input_ids"].to(device),
            attention_mask=train_encodings["attention_mask"].to(device),
            labels=torch.tensor(train_labels).to(device),
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=test_encodings["input_ids"].to(device),
            attention_mask=test_encodings["attention_mask"].to(device),
        )
        probs = softmax(outputs.logits, dim=1)[:, 1]
        preds = (probs > 0.5).int()

    acc_list.append(accuracy_score(test_labels, preds.cpu()))
    f1_list.append(f1_score(test_labels, preds.cpu()))
    all_y_true.extend(test_labels)
    all_y_pred_proba.extend(probs.cpu().numpy())

np.save("distilbert_y_true.npy", np.array(all_y_true))
np.save("distilbert_y_pred_proba.npy", np.array(all_y_pred_proba))

with open("distilbert_results.json", "w") as f:
    json.dump(
        {
            "model": "DistilBERT",
            "accuracy": float(np.mean(acc_list)),
            "f1": float(np.mean(f1_list)),
        },
        f,
    )

model.save_pretrained("distilbert_final_model")
print("✅ DistilBERT Training Complete.")
