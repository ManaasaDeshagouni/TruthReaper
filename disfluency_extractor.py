import warnings
import logging
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Suppress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load Whisper model locally
model_dir = "models/whisper-base"
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

def extract_disfluency_features(audio_path, max_len=300):
    print(f"📝 Advanced disfluency extraction: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return {
            "disfluency_seq": [0] * max_len,
            "total_disfluencies": 0,
            "filler_count": 0,
            "stutter_count": 0,
            "phrase_repetition": 0,
            "disfluency_rate": 0
        }

    # --- Disfluency Detection ---
    filler_pattern = r'\b(um+|uh+|ah+|er+|like|you know|i mean|so|well|hmm|uhh|yeah)\b'
    fillers = re.findall(filler_pattern, transcription)

    stutters = re.findall(r'\b(\w+)( \1){1,}\b', transcription)
    phrase_repeats = re.findall(r'\b(\w+\s\w+)\s+\1\b', transcription)

    total_disfluencies = len(fillers) + len(stutters) + len(phrase_repeats)

    base = total_disfluencies // max_len
    remainder = total_disfluencies % max_len
    disfluency_seq = [base] * max_len
    for i in range(remainder):
        disfluency_seq[i] += 1

    word_count = len(transcription.split())
    disfluency_rate = total_disfluencies / word_count if word_count > 0 else 0

    print(f"✅ Disfluencies Detected: {total_disfluencies} (Fillers: {len(fillers)}, Stutters: {len(stutters)}, Phrases: {len(phrase_repeats)})")

    return {
        "disfluency_seq": disfluency_seq,
        "total_disfluencies": total_disfluencies,
        "filler_count": len(fillers),
        "stutter_count": len(stutters),
        "phrase_repetition": len(phrase_repeats),
        "disfluency_rate": disfluency_rate
    }

if __name__ == "__main__":
    seq = extract_disfluency_features("clips/deception/trial_lie_001.wav")
    print(seq)