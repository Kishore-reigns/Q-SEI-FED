import torch
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YourModelClass(...)   # SAME architecture
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval().to(device)

def extract_embeddings(dataloader):
    embs, labels, known_flags = [], [], []

    with torch.no_grad():
        for x, y, is_known in tqdm(dataloader):
            x = x.to(device)
            z = model(x)          # embedding / logits
            embs.append(z.cpu().numpy())
            labels.append(y.numpy())
            known_flags.append(is_known.numpy())

    return (
        np.concatenate(embs),
        np.concatenate(labels),
        np.concatenate(known_flags)
    )
