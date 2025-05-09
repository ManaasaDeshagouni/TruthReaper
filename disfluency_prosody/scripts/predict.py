import torch
import whisper
import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf
import re
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertModel,
)
from torch.nn.functional import softmax

TRANSCRIPT_MODEL = "models/distilbert_final"
XGB_MODEL_PATH = "xgboost_model.pkl"
TEMP_WAV = "temp_input.wav"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILLERS = ["um", "uh", "erm", "ah"]
REPETITION_PATTERN = re.compile(r"(\b\w+)( \1\b)+")
INCOMPLETE_SENTENCE_PATTERN = re.compile(r"\.\.\.|--|\bbut\b$|\band\b$")


def record_audio(filename=TEMP_WAV, duration=15, samplerate=16000):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved to {filename}")


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
    return probs.argmax(), probs


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

    extra_audio = [np.mean(zcr), np.mean(centroid), np.std(centroid)]

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


def predict_stacked(audio_path):
    transcript = transcribe_audio(audio_path)
    print(f"📝 Transcript:\n{transcript}")

    # DistilBERT
    t_label, t_probs = predict_distilbert(transcript)
    print(f"🤖 DistilBERT → {t_label} with probs {t_probs}")

    # XGBoost
    cls_vec = extract_cls_vector(transcript)
    disfluency, prosody = extract_audio_features(audio_path, transcript)
    full_vec = np.concatenate(
        [cls_vec, disfluency, prosody]
    )  # 768 + 3 + 5 = 776 features

    xgb = joblib.load(XGB_MODEL_PATH)
    a_probs = xgb.predict_proba(full_vec.reshape(1, -1))[0]
    a_label = int(a_probs.argmax())
    print(f"🔊 XGBoost → {a_label} with probs {a_probs}")

    # Fusion (confidence average)
    fused_probs = (t_probs + a_probs) / 2
    final_label = int(fused_probs.argmax())

    print(
        f"\n📊 FINAL STACKED PREDICTION: {'🟥 Deceptive' if final_label else '✅ Truthful'}"
    )
    print(f"    Fused Confidence: {fused_probs}")


if __name__ == "__main__":
    record_audio()
    predict_stacked(TEMP_WAV)
