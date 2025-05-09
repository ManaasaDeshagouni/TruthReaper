import torch
import whisper
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import re
from transformers import DistilBertTokenizer, DistilBertModel

TRANSCRIPT_MODEL = "models/distilbert_final"
TEMP_WAV = "temp_debug.wav"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILLERS = ["um", "uh", "erm", "ah"]
REPETITION_PATTERN = re.compile(r"(\b\w+)( \1\b)+")
INCOMPLETE_SENTENCE_PATTERN = re.compile(r"\.\.\.|--|\bbut\b$|\band\b$")


def record_audio(filename=TEMP_WAV, duration=5, samplerate=16000):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Audio saved to {filename}")


def transcribe_audio(path):
    model = whisper.load_model("medium")
    result = model.transcribe(path, language="en")
    return result["text"]


def extract_cls_vector(text):
    tokenizer = DistilBertTokenizer.from_pretrained(TRANSCRIPT_MODEL)
    model = DistilBertModel.from_pretrained(TRANSCRIPT_MODEL).to(DEVICE)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    print(f"\n🧠 BERT CLS Vector (768-dim): {cls_vector[:5]}...")
    return cls_vector


def extract_audio_features(path, transcript):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    pitch = librosa.yin(y, fmin=75, fmax=500, sr=sr)

    mfcc_stats = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.min(mfcc, axis=1),
            np.max(mfcc, axis=1),
        ]
    )
    print(f"🎵 MFCC Stats Shape: {mfcc_stats.shape}, Example: {mfcc_stats[:5]}")

    extra_audio = [np.mean(zcr), np.mean(centroid), np.std(centroid)]
    print(f"🎛️ Audio Stats: ZCR={extra_audio[0]:.4f}, Centroid={extra_audio[1]:.2f}")

    tokens = transcript.lower().split()
    filler_count = sum(tokens.count(f) for f in FILLERS)
    repetition_count = len(REPETITION_PATTERN.findall(transcript.lower()))
    incomplete_count = len(INCOMPLETE_SENTENCE_PATTERN.findall(transcript.strip()))
    print(
        f"🗣️ Disfluencies → Fillers: {filler_count}, Repetitions: {repetition_count}, Incompletes: {incomplete_count}"
    )

    prosody = [
        np.mean(rms),
        np.std(rms),
        np.mean(pitch),
        np.std(pitch),
        np.sum(zcr < 0.01) / len(zcr),
    ]
    print(f"🔊 Prosody → Energy: {prosody[0]:.4f}, Pitch: {prosody[2]:.2f}")


def debug_from_mic():
    record_audio()
    transcript = transcribe_audio(TEMP_WAV)
    print(f"\n📝 Transcript:\n{transcript}")
    extract_cls_vector(transcript)
    extract_audio_features(TEMP_WAV, transcript)


if __name__ == "__main__":
    debug_from_mic()
