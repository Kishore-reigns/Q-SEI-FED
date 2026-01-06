import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Model.LSNet import LSNet
print("Using LSNet model from Model/LSNet.py")
from Evaluation_metrics.load_model import load_lsnet, extract_embeddings
from Evaluation_metrics.pair_based_conf import pairwise_confusion
from Evaluation_metrics.roc_auroc import roc_unknown_detection
from Evaluation_metrics.eer import compute_eer
from Evaluation_metrics.visualize_embeddings import visualize_embeddings

from torch.utils.data import DataLoader

from Model.TripletDataset import TripletSEIDataset as SEIDataset
print("Imports successful.")
# Load data
dataset = SEIDataset("F:\\K DRIVE\\MIT_Learnings\\Sem8\\dataset\\code\\sei_dataset")
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print("DataLoader created.")
# Load model
model = load_lsnet(
    LSNet,
    os.path.join(os.path.dirname(__file__), "..", "PTH", "lsnet_epoch_50.pth")
)

print("Model loaded.")
# Extract embeddings
embs, labels, known = extract_embeddings(model, loader)
print("Embeddings extracted.")
# Compute class centers (mean of known embeddings)
class_centers = [
    embs[labels == c].mean(axis=0)
    for c in sorted(set(labels[known == 1]))
]

# Pair-based confusion
cm = pairwise_confusion(embs[known == 1], labels[known == 1], threshold=1.0)
print("Verification Confusion Matrix:\n", cm)

# ROC + AUROC
fpr, tpr, auroc = roc_unknown_detection(embs, known, class_centers)
print("AUROC:", auroc)

# EER
eer = compute_eer(fpr, tpr)
print("EER:", eer)

# Visualization
visualize_embeddings(embs[known == 1], labels[known == 1], class_centers)
