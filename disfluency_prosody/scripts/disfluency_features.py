import os
import re
import numpy as np
from tqdm import tqdm

FILLERS = ["um", "uh", "erm", "ah"]
REPETITION_PATTERN = re.compile(r"(\b\w+)( \1\b)+")
INCOMPLETE_SENTENCE_PATTERN = re.compile(r"\.\.\.|--|\bbut\b$|\band\b$")

TRANSCRIPT_ROOT = "../../RLT/Transcription"
SAVE_PATH = "../results/disfluency_features.npy"


def extract_disfluency_features(text):
    tokens = text.lower().split()
    filler_count = sum(tokens.count(f) for f in FILLERS)
    repetition_count = len(REPETITION_PATTERN.findall(text.lower()))
    incomplete_count = len(INCOMPLETE_SENTENCE_PATTERN.findall(text.strip()))
    return [filler_count, repetition_count, incomplete_count]


def process_all_transcripts():
    data = {}
    total, skipped = 0, 0

    for label in ["Deceptive", "Truthful"]:
        path = os.path.join(TRANSCRIPT_ROOT, label)
        for fname in tqdm(os.listdir(path), desc=f"Processing {label}"):
            if not fname.endswith(".txt"):
                continue
            clip_id = fname.replace(".txt", "")
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                text = f.read()

            features = extract_disfluency_features(text)
            if isinstance(features, list) and len(features) == 3:
                data[clip_id] = np.array(features, dtype=np.float32)
                total += 1
            else:
                print(f"⚠️ Skipping {clip_id}: invalid feature shape")
                skipped += 1

    np.save(SAVE_PATH, data)
    print(f"\n✅ Saved {total} disfluency vectors to {SAVE_PATH}. Skipped: {skipped}")


if __name__ == "__main__":
    process_all_transcripts()
