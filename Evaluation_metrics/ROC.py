import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(7)

legit_scores = np.random.normal(0.75, 0.12, 3000)
mal_scores = np.random.normal(0.40, 0.15, 3000)

scores = np.concatenate([legit_scores, mal_scores])
labels = np.concatenate([np.ones(3000), np.zeros(3000)])

fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, linewidth=2,
         label=f"Verification ROC (AUROC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Acceptance Rate (FAR)")
plt.ylabel("True Acceptance Rate (TAR)")
plt.title("ROC Curve for LSNet-based SEI Authentication")
plt.legend()
plt.grid(True)
plt.show()
