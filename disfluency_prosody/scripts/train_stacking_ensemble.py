import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

# Load all features
bert = np.load("../../results/text_bert_embeddings.npy", allow_pickle=True).item()
mfcc = np.load("../results/audio_features_hybrid.npy", allow_pickle=True).item()
disf = np.load("../results/disfluency_features.npy", allow_pickle=True).item()
pros = np.load("../results/prosodic_audio_features.npy", allow_pickle=True).item()

X, y = [], []
for clip_id in bert:
    if clip_id in mfcc and clip_id in disf and clip_id in pros:
        try:
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
        except Exception as e:
            print(f"⚠️ Skipping {clip_id}: {e}")

X = np.array(X)
y = np.array(y)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train base models
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

    # Base model prediction probabilities
    rf_train_probs = rf.predict_proba(X_train)[:, 1].reshape(-1, 1)
    xgb_train_probs = xgb.predict_proba(X_train)[:, 1].reshape(-1, 1)
    meta_train = np.hstack([rf_train_probs, xgb_train_probs])

    rf_test_probs = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)
    xgb_test_probs = xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
    meta_test = np.hstack([rf_test_probs, xgb_test_probs])

    # Meta-classifier (stacking model)
    meta_model = LogisticRegression()
    meta_model.fit(meta_train, y_train)
    final_preds = meta_model.predict(meta_test)

    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    acc_list.append(acc)
    f1_list.append(f1)

    print(
        classification_report(
            y_test, final_preds, target_names=["truthful", "deceptive"]
        )
    )
    print(f"✅ Stacked Accuracy: {acc:.2f}, F1: {f1:.2f}")

# Final results
print("\n🔚 Final Stacking Ensemble Avg")
print(f"✅ Accuracy: {np.mean(acc_list):.2f}, F1: {np.mean(f1_list):.2f}")
with open("stack.json", "w") as f:
    json.dump(
        {
            "model": "stack",
            "accuracy": np.mean(acc_list),
            "f1": np.mean(f1_list),
        },
        f,
    )
