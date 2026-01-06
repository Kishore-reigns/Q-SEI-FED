import torch

from file_picker import pick_npz_file
from data_loader import load_iq_npz
from visualize_iq import plot_iq, plot_constellation
from STFT import iq_to_spectrogram
from visualize_spec import plot_spectrogram
from embedding import get_embedding
from gate import hardware_identity_check
from model import LSNet


def main():
    print("\n=== PHASE-1 : PHYSICAL LAYER SEI DEFENCE ===\n")

    # -------------------------
    # 1. Select dataset
    # -------------------------
    npz_path = pick_npz_file()
    print("[INFO] Selected file:", npz_path)

    # -------------------------
    # 2. Load data
    # -------------------------
    X, y = load_iq_npz(npz_path)

    ref_iq  = X[0]
    test_iq = X[10]

    # -------------------------
    # 3. Visualize IQ
    # -------------------------
    plot_iq(ref_iq, "Reference UAV IQ")
    plot_constellation(ref_iq)

    # -------------------------
    # 4. Spectrogram
    # -------------------------
    spec = iq_to_spectrogram(ref_iq)
    plot_spectrogram(spec)

    # -------------------------
    # 5. Load LSNet
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    model = LSNet(embedding_dim=128)
    model.load_state_dict(torch.load("models/lsnet_triplet.pth", map_location=device))
    model.to(device)
    model.eval()

    # -------------------------
    # 6. Embeddings
    # -------------------------
    z_ref  = get_embedding(ref_iq, model, device)
    z_test = get_embedding(test_iq, model, device)

    # -------------------------
    # 7. Hardware Identity Gate
    # -------------------------
    decision, dist = hardware_identity_check(z_ref, z_test)

    print("\n========== RESULT ==========")
    print("Decision :", decision)
    print("Distance :", dist)
    print("============================")


if __name__ == "__main__":
    main()
