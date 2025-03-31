import os
import time
import torchaudio
from torchaudio.transforms import Resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Suppress warnings
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize model
model_dir = "models/whisper-base"
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

recordings_dir = "recordings"
os.makedirs(recordings_dir, exist_ok=True)

def record_realtime(filename="recordings/realtime.mp3", max_duration=120, silence_limit=15):
    import pyaudio
    import wave
    import audioop

    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"🎙️ Recording live (max {max_duration} sec, stops if {silence_limit} sec of silence)...")
    frames = []
    silence_chunks = 0
    silence_threshold = 500  # Lower = more sensitive to quiet
    max_silence_chunks = int(rate / chunk * silence_limit)
    max_chunks = int(rate / chunk * max_duration)

    for i in range(0, max_chunks):
        data = stream.read(chunk)
        frames.append(data)

        rms = audioop.rms(data, 2)  # root mean square of volume
        if rms < silence_threshold:
            silence_chunks += 1
        else:
            silence_chunks = 0

        if silence_chunks > max_silence_chunks:
            print("🛑 Detected 15 seconds of silence. Stopping early...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("✅ Recording saved.")
    return filename

def resample_audio(audio_path):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        print("🔄 Resampling audio to 16kHz...")
        resampler = Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
    return speech_array[0], 16000

def transcribe(audio_tensor, sampling_rate):
    input_features = processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def smart_label_segments(transcript):
    import re
    segments = re.split(r'(\?|\.|!|\n)', transcript)
    sentences = [segments[i] + segments[i+1] for i in range(0, len(segments)-1, 2)]

    labeled = []
    speaker = "OFFICER"

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if s.endswith("?"):
            speaker = "OFFICER"
        else:
            speaker = "ACCUSED"
        labeled.append(f"[{speaker}]: {s}")

    return "\n".join(labeled)

def main():
    print("🧠 Choose mode:")
    print("[1] Real-Time Live Recording (max 2 mins, stops if 15s silence)")
    print("[2] Use Pre-Recorded Interview Audio")
    choice = input(">> ").strip()

    if choice == "1":
        audio_path = record_realtime()
    elif choice == "2":
        filename = input("Enter path to your .wav file: ").strip()
        if not os.path.exists(filename):
            print("❌ File not found.")
            return
        audio_path = filename
    else:
        print("❌ Invalid choice.")
        return

    audio_tensor, sr = resample_audio(audio_path)
    print("🧠 Transcribing...")
    transcript = transcribe(audio_tensor, sr)

    print("📝 Transcript:")
    print(transcript)

    print("🧠 Labeling speakers...")
    labeled = smart_label_segments(transcript)

    with open(os.path.join(recordings_dir, "marked_transcript.txt"), "w") as f:
        f.write(labeled)

    print("✅ Done. Output saved to recordings/marked_transcript.txt")

if __name__ == "__main__":
    main()
