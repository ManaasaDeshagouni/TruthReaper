import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Load true labels and predicted probabilities
y_true = np.load("distilbert_y_true.npy")
y_pred_proba = np.load("distilbert_y_pred_proba.npy")

# Hardcoded metrics
accuracy = 0.8698347107438017
f1 = 0.856056338028169

# ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# PR
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = average_precision_score(y_true, y_pred_proba)

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Bar Plot
axs[0].bar(["Accuracy", "F1 Score"], [accuracy, f1], color=["skyblue", "salmon"])
axs[0].set_ylim(0.7, 1.0)
axs[0].set_title("Accuracy vs F1 Score")
axs[0].grid(axis="y", linestyle="--", alpha=0.7)

# ROC Curve
axs[1].plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
axs[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
axs[1].set_title("ROC Curve")
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate")
axs[1].legend(loc="lower right")
axs[1].grid(True)

# PR Curve
axs[2].plot(recall, precision, color="green", lw=2, label=f"AUC = {pr_auc:.2f}")
axs[2].set_title("Precision-Recall Curve")
axs[2].set_xlabel("Recall")
axs[2].set_ylabel("Precision")
axs[2].legend(loc="lower left")
axs[2].grid(True)

plt.tight_layout()
plt.show()
