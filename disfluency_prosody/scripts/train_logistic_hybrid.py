import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load all feature files
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc = np.load("../results/audio_features_hybrid.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

X, y = [], []
skipped = 0

for clip_id in bert:
    if clip_id in mfcc and clip_id in disf and clip_id in pros:
        bert_vec = bert[clip_id]
        mfcc_vec = mfcc[clip_id]
        disf_vec = disf[clip_id]
        pros_vec = pros[clip_id]

        # Validate dimensions explicitly
        if (
            isinstance(bert_vec, np.ndarray)
            and bert_vec.ndim == 1
            and isinstance(mfcc_vec, np.ndarray)
            and mfcc_vec.ndim == 1
            and mfcc_vec.shape[0] == 55
            and isinstance(disf_vec, np.ndarray)
            and disf_vec.ndim == 1
            and disf_vec.shape[0] == 3
            and isinstance(pros_vec, np.ndarray)
            and pros_vec.ndim == 1
            and pros_vec.shape[0] == 5
        ):
            full_vec = np.concatenate([bert_vec, mfcc_vec, disf_vec, pros_vec])
            X.append(full_vec)
            y.append(1 if "lie" in clip_id else 0)
        else:
            print(f"⚠️ Skipping {clip_id} — feature shape mismatch")
            skipped += 1

print(f"\n✅ Loaded {len(X)} valid samples. Skipped {skipped} malformed entries.\n")

X = np.array(X)
y = np.array(y)

# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    acc_list.append(acc)
    f1_list.append(f1)

    print(f"\n📂 Fold {fold+1}")
    print(classification_report(y_test, preds, target_names=["truthful", "deceptive"]))
    print(f"✅ Accuracy: {acc:.2f}, F1: {f1:.2f}")

# Final Average
print("\n🔚 Final Logistic Regression Avg")
print(f"✅ Accuracy: {np.mean(acc_list):.2f}, F1: {np.mean(f1_list):.2f}")
import json

with open("logistic_results.json", "w") as f:
    json.dump(
        {
            "model": "Logistic Regression",
            "accuracy": np.mean(acc_list),
            "f1": np.mean(f1_list),
        },
        f,
    )
