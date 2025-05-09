import os
import numpy as np
import librosa
from tqdm import tqdm

AUDIO_ROOT = "../../RLT/Clips"
SAVE_PATH = "../results/audio_mfcc_enhanced.npy"


def extract_enhanced_mfcc(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.min(mfcc, axis=1),
                np.max(mfcc, axis=1),
            ]
        )
    except Exception as e:
        print(f"⚠️ Error: {filepath} — {e}")
        return None


def process_all_clips():
    data = {}
    for label in ["Deceptive", "Truthful"]:
        path = os.path.join(AUDIO_ROOT, label)
        for fname in tqdm(os.listdir(path), desc=f"Processing {label}"):
            if not fname.endswith(".mp4"):
                continue
            clip_id = fname.replace(".mp4", "")
            vec = extract_enhanced_mfcc(os.path.join(path, fname))
            if vec is not None:
                data[clip_id] = vec

    np.save(SAVE_PATH, data)
    print(f"✅ Saved enhanced MFCCs to {SAVE_PATH}")


if __name__ == "__main__":
    process_all_clips()
