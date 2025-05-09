import numpy as np

# Load the audio features
audio_feats = np.load("../results/audio_features.npy", allow_pickle=True).item()

# Total entries
print(f"🎧 Total clips with audio features: {len(audio_feats)}")

# Show one sample
sample_clip_id = list(audio_feats.keys())[0]
print(f"\n🎬 Sample clip ID: {sample_clip_id}")

vec = audio_feats[sample_clip_id]
print(f"📐 Feature shape: {vec.shape}")
print(f"\n🔢 First 10 values:\n{vec[:10]}")
