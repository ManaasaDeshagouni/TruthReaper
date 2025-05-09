import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load features
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc = np.load("../results/audio_features_hybrid.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

X, y = [], []
for clip_id in bert:
    if clip_id in mfcc and clip_id in disf and clip_id in pros:
        bert_vec = bert[clip_id]
        mfcc_vec = mfcc[clip_id]
        disf_vec = disf[clip_id]
        pros_vec = pros[clip_id]

        if all(
            isinstance(v, np.ndarray) and v.ndim == 1
            for v in [bert_vec, mfcc_vec, disf_vec, pros_vec]
        ):
            full_vec = np.concatenate([bert_vec, mfcc_vec, disf_vec, pros_vec])
            X.append(full_vec)
            y.append(1 if "lie" in clip_id else 0)

X = np.array(X)
y = np.array(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_probs = rf.predict_proba(X_test)[:, 1]
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    avg_probs = (rf_probs + xgb_probs) / 2
    final_preds = (avg_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    acc_list.append(acc)
    f1_list.append(f1)

    print(
        classification_report(
            y_test, final_preds, target_names=["truthful", "deceptive"]
        )
    )
    print(f"✅ Ensemble Accuracy: {acc:.2f}, F1: {f1:.2f}")

print("\n🔚 Final Ensemble (RF + XGB) Avg")
print(f"✅ Accuracy: {np.mean(acc_list):.2f}, F1: {np.mean(f1_list):.2f}")
import json

with open("ensemble_results.json", "w") as f:
    json.dump(
        {
            "model": "Stacking (RF + XGB)",
            "accuracy": np.mean(acc_list),
            "f1": np.mean(f1_list),
        },
        f,
    )
