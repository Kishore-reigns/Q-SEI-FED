# Evaluation_metrics/roc_auroc.py

import numpy as np
from sklearn.metrics import roc_curve, auc

def roc_unknown_detection(embeddings, known_flags, class_centers):
    scores = []
    gt = []

    for z, is_known in zip(embeddings, known_flags):
        d_min = min(np.linalg.norm(z - c) for c in class_centers)
        scores.append(-d_min)      # higher = more likely known
        gt.append(is_known)

    fpr, tpr, _ = roc_curve(gt, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
