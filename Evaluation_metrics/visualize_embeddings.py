# Evaluation_metrics/visualize_embeddings.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(
    embeddings,
    labels,
    class_centers,
    title="Class Anchor Feature Space"
):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    centers_2d = pca.transform(class_centers)

    plt.figure(figsize=(8, 6))

    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            alpha=0.6,
            label=f"Class {cls}"
        )

    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c="black",
        marker="X",
        s=200,
        label="Class Centers"
    )

    for c in centers_2d:
        circle = plt.Circle(c, radius=0.8, fill=False, linestyle="--")
        plt.gca().add_patch(circle)

    plt.xlabel("Feature Dimension 1")
    plt.ylabel("Feature Dimension 2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
# visualize_embeddings(embs, labels, class_centers)  # Provide actual data here
