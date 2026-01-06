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

BASE_DIR = "sei_dataset_exp2"
SAMPLES_PER_INSTANCE = 350

# ===============================
# UAV INSTANCES
# ===============================
INSTANCES = {
    "DJI_001":        {"manufacturer": "DJI",     "known": True},
    "DJI_002":        {"manufacturer": "DJI",     "known": True},
    "DJI_003":        {"manufacturer": "DJI",     "known": True},
    "FUTABA_T14_001": {"manufacturer": "FUTABA",  "known": True},
    "FUTABA_T7_001":  {"manufacturer": "FUTABA",  "known": True},
    "GRAUPNER_001":   {"manufacturer": "GRAUPNER","known": True},
    "TURNIGY_001":    {"manufacturer": "TURNIGY", "known": True},
    "ENEMY_DJI":      {"manufacturer": "DJI",     "known": False},
    "ENEMY_FUTABA":   {"manufacturer": "FUTABA",  "known": False},
    "UNKNOWN_NOISE":  {"manufacturer": "UNKNOWN", "known": False},
}

os.makedirs(BASE_DIR, exist_ok=True)

# ===============================
# EXPERIMENT-2 RF PROFILES (OVERLAPPING)
# ===============================
RF_PROFILES = {
    "DJI": {
        "cfo_hz": (-8000, -3000),
        "iq_gain": (0.90, 1.00),
        "iq_phase": (-6, -2),
        "pa_alpha": (1.10, 1.25),
        "phase_noise": (5e-4, 1e-3),
        "filter_fc": 0.20
    },
    "FUTABA": {
        "cfo_hz": (-3000, 2000),
        "iq_gain": (1.00, 1.10),
        "iq_phase": (2, 6),
        "pa_alpha": (0.90, 1.00),
        "phase_noise": (3e-4, 6e-4),
        "filter_fc": 0.25
    },
    "GRAUPNER": {
        "cfo_hz": (1000, 6000),
        "iq_gain": (0.95, 1.05),
        "iq_phase": (-3, 3),
        "pa_alpha": (1.00, 1.10),
        "phase_noise": (2e-4, 4e-4),
        "filter_fc": 0.30
    },
    "TURNIGY": {
        "cfo_hz": (5000, 9000),
        "iq_gain": (1.15, 1.30),
        "iq_phase": (8, 14),
        "pa_alpha": (0.75, 0.90),
        "phase_noise": (8e-4, 1.4e-3),
        "filter_fc": 0.16
    }
}

# ===============================
# SEI PARAMETER SAMPLING (WITH DRIFT)
# ===============================
def sample_sei_params(manufacturer):
    if manufacturer not in RF_PROFILES:
        return {
            "cfo_hz": np.random.uniform(-10000, 10000),
            "iq_gain": np.random.uniform(0.7, 1.3),
            "iq_phase": np.random.uniform(-20, 20) * np.pi / 180,
            "phase_noise": np.random.uniform(1e-4, 2e-3),
            "pa_alpha": np.random.uniform(0.6, 1.4),
            "filter_fc": np.random.uniform(0.1, 0.35)
        }

    p = RF_PROFILES[manufacturer]
    return {
        "cfo_hz": np.random.uniform(*p["cfo_hz"]),
        "iq_gain": np.random.uniform(*p["iq_gain"]),
        "iq_phase": np.random.uniform(*p["iq_phase"]) * np.pi / 180,
        "phase_noise": np.random.uniform(*p["phase_noise"]),
        "pa_alpha": np.random.uniform(*p["pa_alpha"]),
        "filter_fc": p["filter_fc"]
    }

SEI_PARAMS = {
    inst: sample_sei_params(meta["manufacturer"])
    for inst, meta in INSTANCES.items()
}

# ===============================
# RF SIGNAL FUNCTIONS
# ===============================
def generate_qpsk(N):
    bits = np.random.randint(0, 2, (N, 2))
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    return symbols / np.sqrt(2)

def apply_cfo(signal, cfo_hz):
    t = np.arange(len(signal)) / FS
    return signal * np.exp(1j * 2 * np.pi * cfo_hz * t)

def apply_iq_imbalance(signal, gain, phase):
    i = np.real(signal)
    q = np.imag(signal)
    q = gain * (q * np.cos(phase) + i * np.sin(phase))
    return i + 1j * q

def apply_phase_noise(signal, std):
    std *= np.random.uniform(0.8, 1.2)   # drift
    noise = np.random.randn(len(signal)) * std
    return signal * np.exp(1j * noise)

def apply_pa_nonlinearity(signal, alpha):
    return np.sign(signal) * (np.abs(signal) ** alpha)

def apply_rf_filter(signal, fc):
    fc = np.clip(fc + np.random.uniform(-0.03, 0.03), 0.08, 0.40)
    b, a = sps.butter(5, fc)
    return sps.lfilter(b, a, signal)

def add_awgn(signal):
    snr_db = np.random.uniform(5, 30)  # fully overlapping SNR
    p = np.mean(np.abs(signal) ** 2)
    noise_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_p / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise, snr_db

# ===============================
# SPECTROGRAM
# ===============================
def iq_to_spectrogram(iq):
    _, _, Sxx = sps.stft(
        iq,
        fs=FS,
        nperseg=N_PER_SEG,
        noverlap=OVERLAP,
        boundary=None
    )
    spec = 20 * np.log10(np.abs(Sxx) + 1e-6)
    spec = np.clip(spec, -80, 0)
    spec = (spec + 80) / 80
    return spec[:SPEC_SIZE, :SPEC_SIZE].astype(np.float32)

# ===============================
# DATASET GENERATION
# ===============================
print("\n[+] Generating EXPERIMENT-2 (Semi-Realistic) Dataset\n")

for instance_id, meta in INSTANCES.items():
    inst_dir = os.path.join(BASE_DIR, instance_id)
    os.makedirs(inst_dir, exist_ok=True)

    sei = SEI_PARAMS[instance_id]
    print(f"→ {instance_id}")

    for i in tqdm(range(SAMPLES_PER_INSTANCE)):
        if instance_id == "UNKNOWN_NOISE":
            iq = np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES)
        else:
            iq = generate_qpsk(N_SAMPLES)
            iq = apply_cfo(iq, sei["cfo_hz"])
            iq = apply_iq_imbalance(iq, sei["iq_gain"], sei["iq_phase"])
            iq = apply_phase_noise(iq, sei["phase_noise"])
            iq = apply_pa_nonlinearity(iq, sei["pa_alpha"])
            iq = apply_rf_filter(iq, sei["filter_fc"])

        iq, snr = add_awgn(iq)
        spec = iq_to_spectrogram(iq)

        np.savez(
            os.path.join(inst_dir, f"sample_{i:06d}.npz"),
            iq=iq.astype(np.complex64),
            spectrogram=spec,
            snr_db=np.float32(snr),
            instance_id=instance_id,
            manufacturer=meta["manufacturer"],
            is_known=meta["known"],
            sei_params=sei
        )

print("\n✔ EXPERIMENT-2 dataset generation complete.")
print(f"✔ Location: {BASE_DIR}")
