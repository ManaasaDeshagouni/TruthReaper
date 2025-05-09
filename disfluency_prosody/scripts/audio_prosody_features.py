import os
import numpy as np
import librosa
from tqdm import tqdm

AUDIO_ROOT = "../../RLT/Clips"
SAVE_PATH = "../results/prosodic_audio_features.npy"


def extract_prosodic_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        pitch = librosa.yin(y, fmin=75, fmax=500, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        vec = np.array(
            [
                np.mean(rms),
                np.std(rms),
                np.mean(pitch),
                np.std(pitch),
                np.sum(zcr < 0.01) / len(zcr),  # pause ratio
            ],
            dtype=np.float32,
        )

        # Validate shape
        if vec.ndim == 1 and vec.shape[0] == 5:
            return vec
        else:
            raise ValueError(f"Invalid shape: {vec.shape}")
    except Exception as e:
        print(f"⚠️ Error: {filepath} — {e}")
        return None


def process_all_audio():
    features = {}
    total, skipped = 0, 0

    for label in ["Deceptive", "Truthful"]:
        path = os.path.join(AUDIO_ROOT, label)
        for fname in tqdm(os.listdir(path), desc=f"Processing {label}"):
            if not fname.endswith(".mp4"):
                continue

            clip_id = fname.replace(".mp4", "")
            filepath = os.path.join(path, fname)
            vec = extract_prosodic_features(filepath)

            if vec is not None:
                features[clip_id] = vec
                total += 1
            else:
                skipped += 1

    os.makedirs("../results", exist_ok=True)
    np.save(SAVE_PATH, features)
    print(f"\n✅ Saved {total} prosodic vectors to {SAVE_PATH}. Skipped: {skipped}")


if __name__ == "__main__":
    process_all_audio()
