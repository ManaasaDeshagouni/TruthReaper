import json
import torch
import torch.nn as nn
import numpy as np
import argparse

MAX_LEN = 300
FEATURE_DIM = 4
LATENT_DIM = 100

# -------- Pad or Truncate Function --------
def pad_or_truncate(seq, target_len=300):
    seq = seq[:target_len]
    if len(seq) < target_len:
        seq = seq + [0.0] * (target_len - len(seq))
    return seq

# -------- Load Dataset --------
def load_sequences(label):
    with open("sequence_dataset.json", "r") as f:
        data = json.load(f)
    sequences = []
    for entry in data:
        if entry["label"] == label:
            pitch = pad_or_truncate(entry["pitch_seq"], MAX_LEN)
            energy = pad_or_truncate(entry["energy_seq"], MAX_LEN)
            hes = pad_or_truncate(entry["hesitation_seq"], MAX_LEN)
            disf = pad_or_truncate(entry["disfluency_seq"], MAX_LEN)
            seq = np.stack([pitch, energy, hes, disf], axis=1)
            sequences.append(seq)
    return torch.tensor(np.array(sequences), dtype=torch.float32)

# -------- Generator --------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_LEN * FEATURE_DIM),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, MAX_LEN, FEATURE_DIM)

# -------- Discriminator --------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(MAX_LEN * FEATURE_DIM, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, MAX_LEN * FEATURE_DIM)
        return self.model(x)

# -------- Training Loop --------
def train_gan(label, epochs=5000):
    real_data = load_sequences(label)
    batch_size = min(16, len(real_data))

    generator = Generator()
    discriminator = Discriminator()

    loss_fn = nn.BCELoss()
    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        idx = np.random.randint(0, len(real_data), batch_size)
        real_batch = real_data[idx]

        z = torch.randn(batch_size, LATENT_DIM)
        fake_batch = generator(z)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator
        d_real = discriminator(real_batch)
        d_fake = discriminator(fake_batch.detach())
        d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

        optim_D.zero_grad()
        d_loss.backward()
        optim_D.step()

        # Generator
        z = torch.randn(batch_size, LATENT_DIM)
        fake_batch = generator(z)
        g_loss = loss_fn(discriminator(fake_batch), real_labels)

        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), f"generator_{label}.pt")
    print(f"✅ Generator for '{label}' saved as generator_{label}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, choices=["truth", "lie"], required=True)
    parser.add_argument('--epochs', type=int, default=5000)
    args = parser.parse_args()

    train_gan(args.label, args.epochs)