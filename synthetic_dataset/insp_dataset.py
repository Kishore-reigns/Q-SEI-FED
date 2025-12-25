import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "sei_dataset"

# ==================================================
# 1. Dataset Summary
# ==================================================
def dataset_summary():
    print("\n========== DATASET SUMMARY ==========")
    total_samples = 0

    for inst in sorted(os.listdir(DATASET_DIR)):
        inst_path = os.path.join(DATASET_DIR, inst)
        if not os.path.isdir(inst_path):
            continue

        files = [f for f in os.listdir(inst_path) if f.endswith(".npz")]
        count = len(files)
        total_samples += count

        print(f"{inst:<20} : {count} samples")

    print("------------------------------------")
    print(f"Total samples        : {total_samples}")
    print("====================================\n")

# ==================================================
# 2. Inspect One Sample (Deep Inspection)
# ==================================================
def inspect_one_sample():
    for inst in sorted(os.listdir(DATASET_DIR)):
        inst_path = os.path.join(DATASET_DIR, inst)
        if not os.path.isdir(inst_path):
            continue

        sample_file = sorted(
            f for f in os.listdir(inst_path) if f.endswith(".npz")
        )[0]

        data = np.load(os.path.join(inst_path, sample_file), allow_pickle=True)

        iq = data["iq"]
        spec = data["spectrogram"]

        print("========== SAMPLE INSPECTION ==========")
        print(f"Instance ID          : {data['instance_id']}")
        print(f"Manufacturer         : {data['manufacturer']}")
        print(f"Known UAV            : {bool(data['is_known'])}")
        print("--------------------------------------")
        print(f"RF IQ shape          : {iq.shape}")
        print(f"RF IQ dtype          : {iq.dtype}")
        print(f"Spectrogram shape    : {spec.shape}")
        print(f"Spectrogram dtype    : {spec.dtype}")
        print(f"LSNet input shape    : (2, {spec.shape[0]}, {spec.shape[1]})")
        print("======================================\n")

        return iq, spec

# ==================================================
# 3. Visualization
# ==================================================
def visualize_sample(iq, spec):
    plt.figure(figsize=(16, 4))

    # I component
    plt.subplot(1, 3, 1)
    plt.plot(np.real(iq[:5000]))
    plt.title("I Component (Real Part)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    # Q component
    plt.subplot(1, 3, 2)
    plt.plot(np.imag(iq[:5000]))
    plt.title("Q Component (Imag Part)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    # Spectrogram
    plt.subplot(1, 3, 3)
    plt.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="jet"
    )
    plt.title("Spectrogram (LSNet Input)")
    plt.xlabel("Time Bins")
    plt.ylabel("Frequency Bins")

    plt.tight_layout()
    plt.show()

# ==================================================
# 4. Triplet Sanity Check
# ==================================================
def triplet_sanity_check():
    print("========== TRIPLET SANITY CHECK ==========")
    instances = [
        inst for inst in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, inst))
    ]

    if len(instances) < 2:
        print("❌ Not enough instances for triplet loss.")
        return

    anchor = instances[0]
    positive = instances[0]
    negative = instances[1]

    print(f"Anchor   instance : {anchor}")
    print(f"Positive instance : {positive}")
    print(f"Negative instance : {negative}")
    print("✔ Triplet-loss sampling is feasible.")
    print("=========================================\n")

# ==================================================
# Run Everything
# ==================================================
if __name__ == "__main__":
    dataset_summary()
    iq, spec = inspect_one_sample()
    visualize_sample(iq, spec)
    triplet_sanity_check()
    print("✅ All inspections and visualizations complete.\n")