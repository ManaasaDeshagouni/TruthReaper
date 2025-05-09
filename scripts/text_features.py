import os
import torch
from transformers import BertTokenizer, BertModel
from parser import build_dataset
from tqdm import tqdm
import numpy as np

# Load BERT model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# For GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_bert_embedding(text):
    """
    Returns BERT [CLS] embedding for a given text.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().cpu().numpy()


def extract_text_features():
    dataset = build_dataset()
    embeddings = {}

    print("🔍 Extracting BERT features for transcripts...\n")
    for sample in tqdm(dataset):
        try:
            with open(sample["transcript_path"], "r") as f:
                text = f.read().strip()

            emb = get_bert_embedding(text)
            embeddings[sample["clip_id"]] = emb
        except Exception as e:
            print(f"⚠️ Failed on {sample['clip_id']}: {e}")

    os.makedirs("../results", exist_ok=True)
    np.save("../results/text_bert_embeddings.npy", embeddings)
    print("\n✅ Saved embeddings to results/text_bert_embeddings.npy")


if __name__ == "__main__":
    extract_text_features()
