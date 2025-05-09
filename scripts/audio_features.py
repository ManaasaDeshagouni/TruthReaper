import os
import numpy as np
import librosa
from parser import build_dataset
from tqdm import tqdm

AUDIO_FEATURES_PATH = "../results/audio_features.npy"


def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        # Aggregate with mean and std
        feature_vector = np.concatenate(
            [
                mfccs.mean(axis=1),
                mfccs.std(axis=1),
                zcr.mean(axis=1),
                centroid.mean(axis=1),
                rms.mean(axis=1),
            ]
        )

        return feature_vector
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None


def extract_audio_features():
    dataset = build_dataset()
    features = {}

    print("🔊 Extracting audio features from video clips...\n")
    for sample in tqdm(dataset):
        clip_id = sample["clip_id"]
        video_path = sample["video_path"]

        # librosa supports direct loading from .mp4
        vec = extract_features(video_path)
        if vec is not None:
            features[clip_id] = vec

    os.makedirs("../results", exist_ok=True)
    np.save(AUDIO_FEATURES_PATH, features)
    print(f"\n✅ Saved to {AUDIO_FEATURES_PATH}")


if __name__ == "__main__":
    extract_audio_features()
