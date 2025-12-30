# Evaluation_metrics/eer.py

import numpy as np

def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx]
    return eer
