import pyaudio
import wave
import numpy as np
import time
import os

def is_silent(data, threshold=500):
    """Check if audio chunk is silent."""
    return max(data) < threshold

def record_audio(filename="recordings/final_audio.wav", max_duration=120, max_silence=15):
    os.makedirs("recordings", exist_ok=True)

    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    silence_threshold = 500  # You can tweak this value if needed

    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"🎙️ Recording started. Max duration: {max_duration} sec. Max pause allowed: {max_silence} sec.")

    frames = []
    silent_chunk_limit = int((rate / chunk) * max_silence)
    total_chunk_limit = int((rate / chunk) * max_duration)

    silent_chunks = 0
    total_chunks = 0

    while total_chunks < total_chunk_limit:
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)
        frames.append(data)

        if is_silent(audio_data, silence_threshold):
            silent_chunks += 1
        else:
            silent_chunks = 0  # Reset when speech is detected

        if silent_chunks >= silent_chunk_limit:
            print(f"⏸️ Silence > {max_silence} sec detected. Stopping early.")
            break

        total_chunks += 1

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"✅ Audio saved to: {filename}")
    return filename

if __name__ == "__main__":
    record_audio()