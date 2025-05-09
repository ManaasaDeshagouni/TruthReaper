import torch
import json
import numpy as np
import argparse

MAX_LEN = 300
FEATURE_DIM = 4
LATENT_DIM = 100

# -------- Generator Architecture --------
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(LATENT_DIM, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, MAX_LEN * FEATURE_DIM),
            torch.nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, MAX_LEN, FEATURE_DIM)

# -------- Behavioral & Summary Enhancer --------
def enhance_and_summarize(entry, label):
    hesitation_seq = np.array(entry["hesitation_seq"])
    disfluency_seq = np.array(entry["disfluency_seq"])
    pitch_seq = np.array(entry["pitch_seq"])
    energy_seq = np.array(entry["energy_seq"])

    # Behavioral Adjustment
    if label == "lie":
        hesitation_seq += np.random.binomial(1, 0.05, size=MAX_LEN)
        disfluency_seq += np.random.binomial(1, 0.07, size=MAX_LEN)
    elif label == "truth":
        disfluency_seq = disfluency_seq * np.random.binomial(1, 0.8, size=MAX_LEN)

    hesitation_seq = hesitation_seq.clip(0, 5)
    disfluency_seq = disfluency_seq.clip(0, 5)

    # Auto-generate Summary Features
    pause_unit = 0.3  # Assume each hesitation unit ~0.3s
    total_pause_time = float(np.sum(hesitation_seq) * pause_unit)
    avg_pause_duration = float(total_pause_time / (np.count_nonzero(hesitation_seq) + 1e-5))
    long_pause_count = int(np.sum(hesitation_seq > 2))

    total_disfluencies = int(np.sum(disfluency_seq))
    disfluency_rate = float(total_disfluencies / MAX_LEN)

    pitch_variance = float(np.var(pitch_seq))
    avg_energy = float(np.mean(energy_seq))
    energy_variance = float(np.var(energy_seq))
    prosodic_activity = pitch_variance * energy_variance

    # Update entry
    entry.update({
        "hesitation_seq": hesitation_seq.tolist(),
        "disfluency_seq": disfluency_seq.tolist(),
        "total_pause_time": total_pause_time,
        "avg_pause_duration": avg_pause_duration,
        "long_pause_count": long_pause_count,
        "total_disfluencies": total_disfluencies,
        "filler_count": total_disfluencies,   # Approximation
        "stutter_count": int(total_disfluencies * 0.1),  # Assume 10% stutters
        "phrase_repetition": int(total_disfluencies * 0.05),  # Assume 5% phrases
        "disfluency_rate": disfluency_rate,
        "pitch_variance": pitch_variance,
        "avg_energy": avg_energy,
        "energy_variance": energy_variance,
        "prosodic_activity": prosodic_activity
    })

    return entry

# -------- Generate Synthetic Data --------
def generate_synthetic(label, count):
    generator = Generator()
    generator.load_state_dict(torch.load(f"generator_{label}.pt"))
    generator.eval()

    synthetic_data = []

    for i in range(count):
        z = torch.randn(1, LATENT_DIM)
        fake_seq = generator(z).detach().numpy()[0]

        entry = {
            "filename": f"synthetic_{label}_{i+1:03d}.wav",
            "label": label,
            "pitch_seq": fake_seq[:,0].tolist(),
            "energy_seq": fake_seq[:,1].tolist(),
            "hesitation_seq": fake_seq[:,2].tolist(),
            "disfluency_seq": fake_seq[:,3].tolist()
        }

        entry = enhance_and_summarize(entry, label)
        synthetic_data.append(entry)

    output_file = f"synthetic_full_{label}.json"
    with open(output_file, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"✅ Generated {count} fully-enhanced synthetic '{label}' samples. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, choices=["truth", "lie"], required=True)
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()

    generate_synthetic(args.label, args.count)