import os
import pandas as pd

# Path to the RLT dataset folder (relative to this script)
DATA_DIR = "../RLT"


def load_gesture_annotations():
    """
    Loads gesture annotations and labels from labels.csv.
    """
    annot_path = os.path.join(DATA_DIR, "Annotation", "labels.csv")

    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Gesture annotation file not found at: {annot_path}")

    df = pd.read_csv(annot_path)
    df.columns = [col.strip() for col in df.columns]  # Clean column names
    df["ClipID"] = df["id"].apply(lambda x: x.replace(".mp4", ""))
    df.set_index("ClipID", inplace=True)
    return df


def build_dataset():
    """
    Builds a dataset by combining clip info, transcripts, gestures, and labels.
    """
    gesture_df = load_gesture_annotations()
    data = []

    for label_dir in ["Deceptive", "Truthful"]:
        clip_folder = os.path.join(DATA_DIR, "Clips", label_dir)
        transcript_folder = os.path.join(DATA_DIR, "Transcription", label_dir)

        for filename in os.listdir(clip_folder):
            if not filename.endswith(".mp4"):
                continue

            clip_id = filename.replace(".mp4", "")
            video_path = os.path.join(clip_folder, filename)
            transcript_path = os.path.join(transcript_folder, clip_id + ".txt")

            if not os.path.exists(transcript_path):
                print(f"⚠️ Skipping {clip_id}: transcript not found.")
                continue

            if clip_id not in gesture_df.index:
                print(f"⚠️ Skipping {clip_id}: gesture annotation missing.")
                continue

            gestures = (
                gesture_df.loc[clip_id].drop(["id", "class"], errors="ignore").to_dict()
            )
            label = gesture_df.loc[clip_id]["class"]

            sample = {
                "clip_id": clip_id,
                "label": label.lower(),  # 'deceptive' or 'truthful'
                "video_path": video_path,
                "transcript_path": transcript_path,
                "gestures": gestures,
            }

            data.append(sample)

    return data


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"\n✅ Loaded {len(dataset)} samples.")
    print("\n🔎 Sample Entry:\n")
    print(dataset[0])
