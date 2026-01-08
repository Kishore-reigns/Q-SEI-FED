import numpy as np
import torch
import STFT

def preprocess_spec(spec):
    # spec: (2, F, T)
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return torch.tensor(spec, dtype=torch.float32)

def get_embedding(iq_sample, model, device="cpu"):
    spec = STFT.iq_to_spectrogram(iq_sample)
    x = preprocess_spec(spec).unsqueeze(0).to(device)  # (1,2,F,T)

    with torch.no_grad():
        z = model(x)

    return z.squeeze(0).cpu()
