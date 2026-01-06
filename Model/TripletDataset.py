# ===== Cell 4 Dataset =====
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class TripletSEIDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.instances = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]

    def __getitem__(self, idx):
        anchor_inst = self.instances[idx % len(self.instances)]
        neg_inst = random.choice(
            [i for i in self.instances if i != anchor_inst]
        )

        def load_sample(inst):
            files = os.listdir(os.path.join(self.root, inst))
            f = random.choice(files)
            data = np.load(os.path.join(self.root, inst, f), allow_pickle=True)
            spec = data["spectrogram"]
            spec = torch.tensor(
                np.stack([spec, spec]),
                dtype=torch.float32
            )
            return spec

        anchor = load_sample(anchor_inst)
        positive = load_sample(anchor_inst)
        negative = load_sample(neg_inst)

        return anchor, positive, negative

    def __len__(self):
        return 3000  # virtual length
