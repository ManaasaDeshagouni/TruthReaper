import os
import numpy as np
import librosa
from parser import build_dataset
from tqdm import tqdm

SAVE_PATH = "../results/audio_mfcc_sequences.npy"


def extract_mfcc_sequence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T  # Shape: [T x 13]
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None


def process_all_clips():
    dataset = build_dataset()
    mfcc_dict = {}

    print("🎧 Extracting MFCC sequences...\n")
    for sample in tqdm(dataset):
        clip_id = sample["clip_id"]
        audio_path = sample["video_path"]
        mfcc_seq = extract_mfcc_sequence(audio_path)
        if mfcc_seq is not None:
            mfcc_dict[clip_id] = mfcc_seq

    os.makedirs("../results", exist_ok=True)
    np.save(SAVE_PATH, mfcc_dict)
    print(f"\n✅ Saved MFCC sequences to {SAVE_PATH}")


if __name__ == "__main__":
    process_all_clips()
