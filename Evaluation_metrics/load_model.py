# Evaluation_metrics/load_model.py

import torch
import numpy as np
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_lsnet(model_class, pth_path):
    model = model_class()
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, dataloader):
    embeddings = []
    labels = []
    known_flags = []

    with torch.no_grad():
        for x, y, is_known in tqdm(dataloader, desc="Extracting embeddings"):
            x = x.to(DEVICE)
            z = model(x)  # embedding / logits

            embeddings.append(z.cpu().numpy())
            labels.append(y.numpy())
            known_flags.append(is_known.numpy())

    return (
        np.concatenate(embeddings),
        np.concatenate(labels),
        np.concatenate(known_flags),
    )
