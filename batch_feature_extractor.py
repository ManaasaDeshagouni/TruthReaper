import os
import json
import argparse
import emotion_analyzer
import pause_anlyzer
import disfluency_extractor

CLIPS_DIR = "clips"
OUTPUT_FILE = "sequence_dataset.json"
MAX_LEN = 300  # Standardized sequence length

def process_clip(audio_path, label):
    # Get features from each analyzer
    emotion_features = emotion_analyzer.analyze_sequence(audio_path, label, max_len=MAX_LEN)
    pause_features = pause_anlyzer.extract_hesitation_features(audio_path, max_len=MAX_LEN)
    disfluency_features = disfluency_extractor.extract_disfluency_features(audio_path, max_len=MAX_LEN)

    # Combine everything into a single dataset entry
    entry = {
        "filename": audio_path,
        "label": label,

        # Sequences for LSTM
        "pitch_seq": emotion_features["pitch_seq"],
        "energy_seq": emotion_features["energy_seq"],
        "hesitation_seq": pause_features["hesitation_seq"],
        "disfluency_seq": disfluency_features["disfluency_seq"],

        # Pause Summary Features
        "total_pause_time": pause_features["total_pause_time"],
        "avg_pause_duration": pause_features["avg_pause_duration"],
        "long_pause_count": pause_features["long_pause_count"],

        # Disfluency Summary Features
        "total_disfluencies": disfluency_features["total_disfluencies"],
        "filler_count": disfluency_features["filler_count"],
        "stutter_count": disfluency_features["stutter_count"],
        "phrase_repetition": disfluency_features["phrase_repetition"],
        "disfluency_rate": disfluency_features["disfluency_rate"],

        # Emotion/Prosody Summary Features
        "pitch_variance": emotion_features["pitch_variance"],
        "avg_energy": emotion_features["avg_energy"],
        "energy_variance": emotion_features["energy_variance"],
        "prosodic_activity": emotion_features["prosodic_activity"]
    }

    return entry

def batch_extract(limit=10):
    dataset = []
    categories = {"deception": "lie", "truthful": "truth"}

    for folder, label in categories.items():
        path = os.path.join(CLIPS_DIR, folder)
        files = [f for f in os.listdir(path) if f.endswith(".wav")]
        files = files[:limit]

        print(f"\n🔍 Processing {len(files)} files from '{folder}' as '{label}'")

        for fname in files:
            audio_path = os.path.join(path, fname)
            try:
                entry = process_clip(audio_path, label)
                dataset.append(entry)
            except Exception as e:
                print(f"⚠️ Skipped {fname}: {e}")

    # Save dataset
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Feature extraction complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10, help='Max files per category to process')
    args = parser.parse_args()

    batch_extract(limit=args.limit)