import librosa
import numpy as np

def extract_voice_features(audio_file):
    y, sr = librosa.load(audio_file)

    # Extract pitch
    pitch = librosa.yin(y, fmin=80, fmax=400)
    
    # Speech speed detection
    tempo = librosa.beat.tempo(y, sr=sr)
    
    # Zero-crossing rate (voice tremor detection)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return {
        "avg_pitch": np.mean(pitch),
        "speech_speed": tempo[0],
        "voice_tremor": zcr
    }

if __name__ == "__main__":
    features = extract_voice_features("recordings/sample_audio.wav")
    print("🎭 Extracted Features:", features)