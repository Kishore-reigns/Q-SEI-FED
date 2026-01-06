# Evaluation_metrics/pair_based_conf.py

import numpy as np
from sklearn.metrics import confusion_matrix

def pairwise_confusion(embeddings, labels, threshold):
    y_true, y_pred = [], []

    N = len(embeddings)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            same = int(labels[i] == labels[j])
            pred = int(d < threshold)

            y_true.append(same)
            y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred)
    return cm

# Example usage:
# cm = pairwise_confusion(embs, labels, threshold=0.5)  # Adjust threshold as needed   