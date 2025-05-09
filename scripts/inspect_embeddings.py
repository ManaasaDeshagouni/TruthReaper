import numpy as np

# Load the saved embeddings
embeddings = np.load("../results/text_bert_embeddings.npy", allow_pickle=True).item()

# Print total samples
print(f"Total embeddings: {len(embeddings)}")

# Pick a sample
sample_clip_id = list(embeddings.keys())[0]
print(f"\n🎬 Sample clip ID: {sample_clip_id}")

# View shape of the vector
vector = embeddings[sample_clip_id]
print(f"📐 Embedding shape: {vector.shape}")  # should be (768,)

# View first 10 values of the vector
print(f"\n🔢 First 10 values:\n{vector[:10]}")
