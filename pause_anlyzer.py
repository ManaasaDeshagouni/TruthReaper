import librosa
import numpy as np

def extract_hesitation_features(audio_path, threshold=0.01, frame_length=2048, hop_length=512, pause_threshold_sec=0.5, max_len=300):
    print(f"\n⏸ Extracting hesitation features from: {audio_path}")

    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return {}

    # Calculate RMS energy across frames
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Frame-level hesitation sequence (binary)
    hesitation_seq = (rms < threshold).astype(int).tolist()

    # Calculate pause durations in seconds
    pause_durations = []
    current_pause = 0

    for i in range(len(rms)):
        if rms[i] < threshold:
            current_pause += hop_length / sr
        elif current_pause > 0:
            pause_durations.append(current_pause)
            current_pause = 0
    if current_pause > 0:
        pause_durations.append(current_pause)

    total_pause_time = sum(pause_durations)
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    long_pause_count = sum(1 for p in pause_durations if p >= pause_threshold_sec)

    return {
        "hesitation_seq": hesitation_seq[:max_len],
        "total_pause_time": total_pause_time,
        "avg_pause_duration": avg_pause_duration,
        "long_pause_count": long_pause_count
    }