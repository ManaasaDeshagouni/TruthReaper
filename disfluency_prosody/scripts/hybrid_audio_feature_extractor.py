import os
import numpy as np
import librosa
from tqdm import tqdm

# Adjust based on your dataset location
AUDIO_ROOT = "../../RLT/Clips"
SAVE_PATH = "../results/audio_features_hybrid.npy"


def extract_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rms = librosa.feature.rms(y=y)[0]

        vec = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.min(mfcc, axis=1),
                np.max(mfcc, axis=1),
                [np.mean(zcr)],
                [np.mean(centroid)],
                [np.mean(rms)],
            ]
        )

        # Ensure it's a valid 1D vector
        if vec.ndim != 1 or vec.shape[0] != 55:
            raise ValueError(f"Invalid feature shape: {vec.shape}")

        return vec

    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
        return None


def process_all_clips():
    features = {}
    total, skipped = 0, 0

    for label in ["Deceptive", "Truthful"]:
        path = os.path.join(AUDIO_ROOT, label)
        for fname in tqdm(os.listdir(path), desc=f"Processing {label}"):
            if not fname.endswith(".mp4"):
                continue

            clip_id = fname.replace(".mp4", "")
            filepath = os.path.join(path, fname)
            vec = extract_features(filepath)

            if vec is not None:
                features[clip_id] = vec
                total += 1
            else:
                skipped += 1

    print(f"\n✅ Extracted features for {total} clips. Skipped {skipped} malformed.")
    os.makedirs("../results", exist_ok=True)
    np.save(SAVE_PATH, features)
    print(f"📦 Saved hybrid features to: {SAVE_PATH}")


if __name__ == "__main__":
    process_all_clips()
