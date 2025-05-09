import os
import torch
import librosa
import numpy as np
import whisper
import joblib
import re
import subprocess
import matplotlib.pyplot as plt
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
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
)
from torch.nn.functional import softmax

# Paths
TRANSCRIPT_MODEL = "models/distilbert_final"
XGB_MODEL_PATH = "xgboost_model.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Patterns
FILLERS = ["um", "uh", "erm", "ah"]
REPETITION_PATTERN = re.compile(r"(\b\w+)( \1\b)+")
INCOMPLETE_SENTENCE_PATTERN = re.compile(r"\.\.\.|--|\bbut\b$|\band\b$")


# Extract .wav from .mp4
def extract_audio_from_mp4(mp4_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def transcribe_audio(path):
    model = whisper.load_model("medium")
    result = model.transcribe(path, language="en", temperature=0, fp16=False)
    return result["text"]


def predict_distilbert(text):
    tokenizer = DistilBertTokenizer.from_pretrained(TRANSCRIPT_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(TRANSCRIPT_MODEL).to(
        DEVICE
    )
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return probs


def extract_cls_vector(text):
    tokenizer = DistilBertTokenizer.from_pretrained(TRANSCRIPT_MODEL)
    model = DistilBertModel.from_pretrained(TRANSCRIPT_MODEL).to(DEVICE)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def extract_audio_features(path, transcript):
    y, sr = librosa.load(path, sr=None)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    pitch = librosa.yin(y, fmin=75, fmax=500, sr=sr)

    tokens = transcript.lower().split()
    filler_count = sum(tokens.count(f) for f in FILLERS)
    repetition_count = len(REPETITION_PATTERN.findall(transcript.lower()))
    incomplete_count = len(INCOMPLETE_SENTENCE_PATTERN.findall(transcript.strip()))
    disfluency = [filler_count, repetition_count, incomplete_count]

    prosody = [
        np.mean(rms),
        np.std(rms),
        np.mean(pitch),
        np.std(pitch),
        np.sum(zcr < 0.01) / len(zcr),
    ]

    return disfluency, prosody


def predict_fusion(wav_path):
    transcript = transcribe_audio(wav_path)
    cls_vec = extract_cls_vector(transcript)
    disfluency, prosody = extract_audio_features(wav_path, transcript)
    full_vec = np.concatenate([cls_vec, disfluency, prosody])

    t_probs = predict_distilbert(transcript)
    xgb = joblib.load(XGB_MODEL_PATH)
    a_probs = xgb.predict_proba(full_vec.reshape(1, -1))[0]

    fused_probs = (t_probs + a_probs) / 2
    return fused_probs


# 🔎 Prepare test samples from RLT/Clips/
samples = []

for fname in os.listdir("../../RLT/Clips/Deceptive"):
    if fname.endswith(".mp4"):
        samples.append((f"../../RLT/Clips/Deceptive/{fname}", 1))

for fname in os.listdir("../../RLT/Clips/Truthful"):
    if fname.endswith(".mp4"):
        samples.append((f"../../RLT/Clips/Truthful/{fname}", 0))

# 🔍 Run predictions
y_true = []
y_pred = []
y_probs = []

for mp4_path, label in samples:
    wav_path = "temp_eval.wav"
    extract_audio_from_mp4(mp4_path, wav_path)
    probs = predict_fusion(wav_path)
    pred = int(probs.argmax())
    y_true.append(label)
    y_pred.append(pred)
    y_probs.append(probs[1])  # prob of class 1 (deceptive)

# 🧠 Evaluation
print("✅ Evaluation Complete")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Truthful", "Deceptive"]))

# 📊 Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stacked Model ROC Curve")
plt.legend()
plt.savefig("stacked_model_roc_curve.png")

# 📈 PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_probs)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Stacked Model Precision-Recall Curve")
plt.savefig("stacked_model_pr_curve.png")

# 🔲 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Stacked Model Confusion Matrix")
plt.savefig("stacked_model_confusion_matrix.png")
