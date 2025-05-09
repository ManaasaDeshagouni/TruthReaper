import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import joblib
import json

# Load features
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc = np.load("../results/audio_features_hybrid.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

# Combine all features per clip
X, y = [], []
for clip_id in bert:
    if clip_id in mfcc and clip_id in disf and clip_id in pros:
        full_vec = np.concatenate(
            [
                bert[clip_id],  # 768-d
                mfcc[clip_id],  # 55-d
                disf[clip_id],  # 3-d
                pros[clip_id],  # 5-d
            ]
        )
        X.append(full_vec)
        y.append(1 if "lie" in clip_id else 0)

X = np.array(X)
y = np.array(y)

# ✅ Show label distribution
label_dist = Counter(y)
print("✅ Label distribution:", label_dist)

# ⚖️ Compute scale_pos_weight if classes are imbalanced
scale_pos_weight = label_dist[0] / label_dist[1] if label_dist[1] != 0 else 1
print(f"⚖️ scale_pos_weight set to: {scale_pos_weight:.2f}")

# Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # ✅ handle class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    acc_list.append(acc)
    f1_list.append(f1)

    print(classification_report(y_test, preds, target_names=["truthful", "deceptive"]))
    print(f"✅ Accuracy: {acc:.2f}, F1: {f1:.2f}")

# Final summary
avg_acc = np.mean(acc_list)
avg_f1 = np.mean(f1_list)
print("\n🔚 Final Enhanced XGBoost Avg")
print(f"✅ Accuracy: {avg_acc:.2f}, F1: {avg_f1:.2f}")

# Save to JSON
with open("xgboost_results.json", "w") as f:
    json.dump({"model": "XGBoost", "accuracy": float(avg_acc), "f1": float(avg_f1)}, f)

# Save model
joblib.dump(model, "xgboost_model.pkl")
print("✅ Saved XGBoost model to xgboost_model.pkl")
