from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from torchaudio.transforms import Resample
import os
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load model and processor
model_dir = "models/whisper-base"
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)

# OPTIONAL FIX to remove forced decoder errors
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Load the recorded audio
audio_path = "recordings/final_audio.wav"
speech_array, sampling_rate = torchaudio.load(audio_path)

# Resample audio to 16000 Hz if necessary
if sampling_rate != 16000:
    print(f"🔁 Resampling from {sampling_rate} Hz to 16000 Hz")
    resampler = Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_array = resampler(speech_array)
    sampling_rate = 16000

# Convert audio to model input
input_features = processor(speech_array[0], sampling_rate=sampling_rate, return_tensors="pt").input_features

# Transcribe
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Output transcription
print("\n📝 Transcription:")
print(transcription)

# Save transcription to file
os.makedirs("recordings", exist_ok=True)
with open("recordings/transcription.txt", "w") as f:
    f.write(transcription)