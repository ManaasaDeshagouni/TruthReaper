import os
import pandas as pd

# Path to RLT relative from disfluency_prosody/scripts
DATA_DIR = "../../RLT"


def load_gesture_annotations():
    annot_path = os.path.join(DATA_DIR, "Annotation", "labels.csv")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Gesture annotation file not found at: {annot_path}")

    df = pd.read_csv(annot_path)
    df.columns = [col.strip() for col in df.columns]
    df["ClipID"] = df["id"].apply(lambda x: x.replace(".mp4", ""))
    df.set_index("ClipID", inplace=True)
    return df


def build_dataset():
    gesture_df = load_gesture_annotations()
    data = []

    for label in ["Deceptive", "Truthful"]:
        clip_folder = os.path.join(DATA_DIR, "Clips", label)
        transcript_folder = os.path.join(DATA_DIR, "Transcription", label)

        for fname in os.listdir(clip_folder):
            if not fname.endswith(".mp4"):
                continue

            clip_id = fname.replace(".mp4", "")
            video_path = os.path.join(clip_folder, fname)
            transcript_path = os.path.join(transcript_folder, clip_id + ".txt")

            if not os.path.exists(transcript_path):
                continue
            if clip_id not in gesture_df.index:
                continue

            gestures = (
                gesture_df.loc[clip_id].drop(["id", "class"], errors="ignore").to_dict()
            )
            label_val = gesture_df.loc[clip_id]["class"]

            sample = {
                "clip_id": clip_id,
                "label": label_val.lower(),
                "video_path": video_path,
                "transcript_path": transcript_path,
                "gestures": gestures,
            }
            data.append(sample)

    return data


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"✅ Loaded {len(dataset)} samples.")
    print("🔎 Sample:", dataset[0])
