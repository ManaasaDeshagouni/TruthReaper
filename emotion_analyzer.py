import librosa
import numpy as np
import os
import json

SEQUENCE_DATASET = "sequence_dataset.json"

def extract_pitch_sequence(y, sr, hop_length=512):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_seq = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_seq.append(float(pitch))
        else:
            pitch_seq.append(0.0)
    
    return pitch_seq

def extract_energy_sequence(y, hop_length=512):
    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return [float(e) for e in energy]

def save_sequence_to_dataset(filename, pitch_seq, energy_seq, label):
    entry = {
        "filename": filename,
        "label": label,
        "pitch_seq": pitch_seq,
        "energy_seq": energy_seq
    }

    if not os.path.exists(SEQUENCE_DATASET):
        with open(SEQUENCE_DATASET, "w") as f:
            json.dump([entry], f, indent=2)
    else:
        with open(SEQUENCE_DATASET, "r") as f:
            existing = json.load(f)
        existing.append(entry)
        with open(SEQUENCE_DATASET, "w") as f:
            json.dump(existing, f, indent=2)

def analyze_sequence(audio_path, label, max_len=300):
    print(f"\n🎭 Extracting pitch & energy + prosodic features from: {audio_path}")

    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return

    pitch_seq = extract_pitch_sequence(y, sr)
    energy_seq = extract_energy_sequence(y)

    # Pad or truncate
    pitch_seq = pitch_seq[:max_len] + [0.0] * max(0, max_len - len(pitch_seq))
    energy_seq = energy_seq[:max_len] + [0.0] * max(0, max_len - len(energy_seq))

    # --- New Prosodic Summary Features ---
    pitch_array = np.array(pitch_seq)
    energy_array = np.array(energy_seq)

    pitch_variance = float(np.var(pitch_array))
    avg_energy = float(np.mean(energy_array))
    energy_variance = float(np.var(energy_array))
    prosodic_activity = pitch_variance * energy_variance

    return {
        "pitch_seq": pitch_seq,
        "energy_seq": energy_seq,
        "pitch_variance": pitch_variance,
        "avg_energy": avg_energy,
        "energy_variance": energy_variance,
        "prosodic_activity": prosodic_activity
    }

if __name__ == "__main__":
    # Manual test example:
    audio_file = "recordings/q1.wav"
    label = "truth"  # or "lie"
    analyze_sequence(audio_file, label)