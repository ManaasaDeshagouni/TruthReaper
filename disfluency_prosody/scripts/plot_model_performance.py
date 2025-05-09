import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = "./"

# Collect *_results.json files
json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_results.json")]

model_names, accuracies, f1_scores = [], [], []

for file in json_files:
    with open(os.path.join(RESULTS_DIR, file), "r") as f:
        data = json.load(f)
        model_names.append(data.get("model", file.replace("_results.json", "")))
        accuracies.append(data.get("accuracy", 0))
        f1_scores.append(data.get("f1", 0))

# Bar Plot of Accuracy and F1
x = range(len(model_names))
bar_width = 0.35

plt.figure(figsize=(12, 6))
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
plt.title("Model Performance Comparison (Accuracy & F1)")
plt.xticks([i + bar_width / 2 for i in x], model_names, rotation=30, ha="right")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("model_accuracy_f1_comparison.png")
plt.show()

print("✅ Plotted Accuracy/F1 bar chart.")

# Plot ROC, PR, Confusion Matrix if y_true/y_pred_proba exist
for model in model_names:
    safe_name = model.lower().replace(" ", "_")
    true_path = f"{safe_name}_y_true.npy"
    proba_path = f"{safe_name}_y_pred_proba.npy"

    if os.path.exists(true_path) and os.path.exists(proba_path):
        print(f"📊 Found predictions for {model} - plotting advanced metrics...")
        y_true = np.load(true_path)
        y_pred_proba = np.load(proba_path)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{safe_name}_roc_curve.png")
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {model}")
        plt.grid()
        plt.savefig(f"{safe_name}_pr_curve.png")
        plt.show()

        # Confusion Matrix
        preds = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {model}")
        plt.savefig(f"{safe_name}_confusion_matrix.png")
        plt.show()

    else:
        print(f"⚠️ Missing y_true/y_pred_proba for {model}, skipping ROC/PR/ConfMat.")
