import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01   # Adjust if needed
SILENCE_DURATION = 10      # Seconds of silence to stop recording
CLIP_FOLDER = "clips/truthful"

def get_next_filename():
    existing = [f for f in os.listdir(CLIP_FOLDER) if f.startswith("trial_truth_custom") and f.endswith(".wav")]
    next_num = len(existing) + 1
    return f"trial_truth_{next_num:03d}.wav"

def record_audio():
    print("🎤 Start speaking... Recording will auto-stop after 10 seconds of silence.")
    recording = []
    silence_counter = 0
    block_duration = 0.5  # seconds

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
        while True:
            block = stream.read(int(SAMPLE_RATE * block_duration))[0]
            rms = np.sqrt(np.mean(block**2))
            recording.append(block)

            if rms < SILENCE_THRESHOLD:
                silence_counter += block_duration
            else:
                silence_counter = 0

            if silence_counter >= SILENCE_DURATION:
                print("⏹️  Detected silence. Stopping recording.")
                break

    audio = np.concatenate(recording, axis=0)
    filename = get_next_filename()
    wav.write(os.path.join(CLIP_FOLDER, filename), SAMPLE_RATE, audio)
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    record_audio()