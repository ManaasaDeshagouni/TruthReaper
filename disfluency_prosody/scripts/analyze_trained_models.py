import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Config
RESULTS_DIR = "./"

# Models you trained
model_keys = ["lstm_fusion", "xgboost", "distilbert"]

# Accuracy/F1 bar plot
model_names, accuracies, f1_scores = [], [], []

for key in model_keys:
    with open(f"{key}_results.json", "r") as f:
        res = json.load(f)
        model_names.append(res.get("model", key))
        accuracies.append(res.get("accuracy", 0))
        f1_scores.append(res.get("f1", 0))

# Bar Chart
x = range(len(model_names))
bar_width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x, accuracies, width=bar_width, label="Accuracy", color="skyblue")
plt.bar(
    [i + bar_width for i in x],
    f1_scores,
    width=bar_width,
    label="F1 Score",
    color="orange",
)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Accuracy vs F1 Score Comparison")
plt.xticks([i + bar_width / 2 for i in x], model_names, rotation=30, ha="right")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_f1_comparison.png")
plt.show()

# ROC, PR Curve, Confusion Matrix per model
for key in model_keys:
    y_true = np.load(f"{key}_y_true.npy")
    y_pred_proba = np.load(f"{key}_y_pred_proba.npy")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {key}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{key}_roc_curve.png")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {key}")
    plt.grid()
    plt.savefig(f"{key}_pr_curve.png")
    plt.show()

    # Confusion Matrix
    preds = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {key}")
    plt.savefig(f"{key}_confusion_matrix.png")
    plt.show()
