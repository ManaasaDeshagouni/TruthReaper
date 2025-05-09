import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load saved predictions
y_true = np.load("distilbert_y_true.npy")
y_pred_proba = np.load("distilbert_y_pred_proba.npy")

# --------------------------------
# 📈 ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DistilBERT ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("distilbert_roc_curve.png")
plt.show()

# --------------------------------
# 📉 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DistilBERT Precision-Recall Curve")
plt.grid()
plt.tight_layout()
plt.savefig("distilbert_pr_curve.png")
plt.show()

# --------------------------------
# 🔲 Confusion Matrix
preds = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("DistilBERT Confusion Matrix")
plt.tight_layout()
plt.savefig("distilbert_confusion_matrix.png")
plt.show()
