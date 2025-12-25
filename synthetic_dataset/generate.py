import numpy as np
import scipy.signal as sps
import os
from tqdm import tqdm

# ===============================
# Global Parameters
# ===============================
FS = 1_000_000
N_SAMPLES = 200_000
N_PER_SEG = 1024
OVERLAP = 512
SPEC_SIZE = 1024

BASE_DIR = "sei_dataset"
SAMPLES_PER_INSTANCE = 500

# ===============================
# UAV INSTANCES (EDITABLE)
# ===============================
INSTANCES = {
    # -------- OUR UAVs (KNOWN) --------
    "DJI_001":        {"manufacturer": "DJI",     "known": True},
    "DJI_002":        {"manufacturer": "DJI",     "known": True},
    "DJI_003":        {"manufacturer": "DJI",     "known": True},
    "FUTABA_T14_001": {"manufacturer": "FUTABA",  "known": True},
    "FUTABA_T7_001":  {"manufacturer": "FUTABA",  "known": True},
    "GRAUPNER_001":   {"manufacturer": "GRAUPNER","known": True},
    "TURNIGY_001":    {"manufacturer": "TURNIGY", "known": True},

    # -------- ADVERSARIAL / UNKNOWN --------
    "ENEMY_DJI":      {"manufacturer": "DJI",     "known": False},
    "ENEMY_FUTABA":   {"manufacturer": "FUTABA",  "known": False},
    "UNKNOWN_NOISE":  {"manufacturer": "UNKNOWN", "known": False},
}

os.makedirs(BASE_DIR, exist_ok=True)

# ===============================
# RF SIGNAL GENERATION
# ===============================
def generate_qpsk(N):
    bits = np.random.randint(0, 2, (N, 2))
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    return symbols / np.sqrt(2)

def add_awgn(signal, snr_db):
    p = np.mean(np.abs(signal) ** 2)
    noise_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_p / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise

def iq_to_spectrogram(iq):
    _, _, Sxx = sps.stft(
        iq,
        fs=FS,
        nperseg=N_PER_SEG,
        noverlap=OVERLAP,
        boundary=None
    )
    spec = np.log1p(np.abs(Sxx))
    return spec[:SPEC_SIZE, :SPEC_SIZE].astype(np.float32)

# ===============================
# DATASET GENERATION
# ===============================
print("\n[+] Generating SEI Dataset (Instance-Level, Triplet-Ready)\n")

for instance_id, meta in INSTANCES.items():
    inst_dir = os.path.join(BASE_DIR, instance_id)
    os.makedirs(inst_dir, exist_ok=True)

    print(f"→ Generating {SAMPLES_PER_INSTANCE} samples for {instance_id}")

    for i in tqdm(range(SAMPLES_PER_INSTANCE)):
        if instance_id == "UNKNOWN_NOISE":
            iq = np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES)
        else:
            iq = generate_qpsk(N_SAMPLES)

        iq = add_awgn(iq, np.random.uniform(-10, 30))
        spec = iq_to_spectrogram(iq)

        np.savez(
            os.path.join(inst_dir, f"sample_{i:06d}.npz"),
            iq=iq.astype(np.complex64),
            spectrogram=spec,
            instance_id=instance_id,
            manufacturer=meta["manufacturer"],
            is_known=meta["known"]
        )

print("\n✔ Dataset generation complete.")
print(f"✔ Location: {BASE_DIR}")
