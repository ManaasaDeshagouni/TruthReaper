import os
import time
import json
import torch
import numpy as np
import torchaudio
import librosa
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pause_anlyzer
import disfluency_extractor
from sequence_lstm_trainer import TruthReaperLSTM, pad_or_truncate, min_max_normalize, normalize_summary_features, MAX_LEN

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


model_dir = "models/whisper-base"
processor = WhisperProcessor.from_pretrained(model_dir)
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_dir)
whisper_model.config.forced_decoder_ids = None
whisper_model.config.suppress_tokens = []

os.makedirs("analysis/reports", exist_ok=True)

# -------- Real-Time Recorder --------
def record_realtime(filename="recordings/realtime.wav", max_duration=120, silence_limit=6):
    import pyaudio, wave, audioop
    chunk, format, channels, rate = 1024, pyaudio.paInt16, 1, 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print(f"🎙️ Recording... (max {max_duration}s, stops if {silence_limit}s silence)")
    frames, silence_chunks = [], 0
    for _ in range(int(rate / chunk * max_duration)):
        data = stream.read(chunk)
        frames.append(data)
        rms = audioop.rms(data, 2)
        silence_chunks = silence_chunks + 1 if rms < 500 else 0
        if silence_chunks > int(rate / chunk * silence_limit):
            print("🛑 Silence detected. Stopping...")
            break
    stream.stop_stream(); stream.close(); p.terminate()
    wf = wave.open(filename, 'wb'); wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format)); wf.setframerate(rate)
    wf.writeframes(b''.join(frames)); wf.close()
    print(f"✅ Saved: {filename}")
    return filename

# -------- Transcription --------
def resample_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return (Resample(sr, 16000)(audio)[0], 16000) if sr != 16000 else (audio[0], sr)

def transcribe(audio_tensor, sr):
    features = processor(audio_tensor, sampling_rate=sr, return_tensors="pt").input_features
    ids = whisper_model.generate(features)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# -------- Pitch & Energy Extraction --------
def extract_pitch_energy(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_seq = [float(pitches[:, t].max()) for t in range(pitches.shape[1])]
    energy_seq = librosa.feature.rms(y=y)[0].tolist()
    pitch_seq = pad_or_truncate(pitch_seq)
    energy_seq = pad_or_truncate(energy_seq)
    return pitch_seq, energy_seq

# -------- Full Feature Extraction --------
def extract_features(audio_path):
    pitch_seq, energy_seq = extract_pitch_energy(audio_path)
    hesitation_data = pause_anlyzer.extract_hesitation_features(audio_path)
    hesitation_seq = list(map(int, hesitation_data["hesitation_seq"]))

    total_pause_time = hesitation_data["total_pause_time"]
    avg_pause_duration = hesitation_data["avg_pause_duration"]
    long_pause_count = hesitation_data["long_pause_count"]

    disfluency_data = disfluency_extractor.extract_disfluency_features(audio_path)
    disfluency_seq = disfluency_data["disfluency_seq"]

    # Auto-summary
    summary_feats = [
        sum(hesitation_seq) * 0.3,
        (sum(hesitation_seq) * 0.3) / (np.count_nonzero(hesitation_seq) + 1e-5),
        int(np.sum(np.array(hesitation_seq) > 2)),
        disfluency_data["total_disfluencies"],
        disfluency_data["filler_count"],
        disfluency_data["stutter_count"],
        disfluency_data["phrase_repetition"],
        disfluency_data["disfluency_rate"],
        np.var(pitch_seq),
        np.mean(energy_seq),
        np.var(energy_seq),
        np.var(pitch_seq) * np.var(energy_seq)
    ]

    pitch = min_max_normalize(pitch_seq)
    energy = min_max_normalize(energy_seq)
    hes = pad_or_truncate(hesitation_seq)
    disf = pad_or_truncate(disfluency_seq)

    seq_tensor = torch.tensor([np.stack([pitch, energy, hes, disf], axis=1)], dtype=torch.float32)
    summary_tensor = normalize_summary_features([summary_feats])

    return seq_tensor, summary_tensor

# -------- Prediction --------
def predict_truth_or_lie(seq_tensor, summary_tensor):
    model = TruthReaperLSTM()
    model.load_state_dict(torch.load("truthreaper_hybrid_lstm.pt"))
    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor, summary_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
    return ("truth", probs[0]) if pred == 0 else ("lie", probs[1])

# -------- Main --------
def main():
    print("🎤 [1] Record")
    choice = input(">> ").strip()
    audio_path = record_realtime() if choice == "1" else input("Enter .wav path: ").strip()

    audio_tensor, sr = resample_audio(audio_path)
    print("📝 Transcribing...")
    transcript = transcribe(audio_tensor, sr)
    print(f"\n📄 Transcript: {transcript}")

    print("⚙️ Extracting features...")
    seq_tensor, summary_tensor = extract_features(audio_path)

    print("🤖 Predicting...")
   
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prediction, confidence = predict_truth_or_lie(seq_tensor, summary_tensor)
    print(f"\n🚨 Prediction: {prediction.upper()} (Confidence: {confidence:.2%})")

    report = {
    "transcript": transcript,
    "prediction": prediction,
    "confidence": float(confidence),   # <-- Fix here
    "audio": audio_path
    }
    with open(f"analysis/reports/session_{timestamp}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📂 Report saved: analysis/reports/session_{timestamp}.json")

if __name__ == "__main__":
    main()