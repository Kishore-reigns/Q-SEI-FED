import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
CSV_FILE = "training_metrics_final4.csv"
OUTPUT_DIR = "plots"

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(CSV_FILE)

epochs = df["epoch"]
triplet_loss = df["triplet_loss"]
pos_dist = df["avg_pos_dist"]
neg_dist = df["avg_neg_dist"]
margin = df["margin"]
accuracy = df["verification_accuracy"]

# ============================
# 1. Triplet Loss vs Epoch
# ============================
plt.figure(figsize=(8, 4))
plt.plot(epochs, triplet_loss, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Triplet Loss")
plt.title("Triplet Loss vs Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/triplet_loss.png", dpi=300)
plt.show()

# ============================
# 2. Positive vs Negative Distance
# ============================
plt.figure(figsize=(8, 4))
plt.plot(epochs, pos_dist, marker="o", label="Avg Positive Distance")
plt.plot(epochs, neg_dist, marker="s", label="Avg Negative Distance")
plt.xlabel("Epoch")
plt.ylabel("Embedding Distance")
plt.title("Positive vs Negative Distance vs Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pos_neg_distance.png", dpi=300)
plt.show()

# ============================
# 3. Margin Evolution
# ============================
plt.figure(figsize=(8, 4))
plt.plot(epochs, margin, marker="^")
plt.xlabel("Epoch")
plt.ylabel("Margin")
plt.title("Triplet Margin Evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/margin.png", dpi=300)
plt.show()

# ============================
# 4. Verification Accuracy
# ============================
plt.figure(figsize=(8, 4))
plt.plot(epochs, accuracy * 100, marker="d")
plt.xlabel("Epoch")
plt.ylabel("Verification Accuracy (%)")
plt.title("Verification Accuracy vs Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/verification_accuracy.png", dpi=300)
plt.show()

print("✅ All plots generated successfully!")
