import torch

import data_loader
from file_picker import pick_npz_file
import visualize_iq 
import STFT
import visualize_spec
import embedding
import gate
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
    iq, meta = data_loader.load_iq_npz(npz_path)

    ref_iq  = iq
    test_iq = iq
    # -------------------------
    # 3. Visualize IQ
    # -------------------------
    visualize_iq.plot_iq(ref_iq, "Reference UAV IQ")
    visualize_iq.plot_constellation(ref_iq)

    # -------------------------
    # 4. Spectrogram
    # -------------------------
    spec = STFT.iq_to_spectrogram(ref_iq)
    visualize_spec.plot_spectrogram(spec)


    spec_combined = STFT.iq_to_combined_spectrogram(ref_iq)

    visualize_spec.plot_combined_spectrogram(spec_combined)

    # -------------------------
    # 5. Load LSNet
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    model = LSNet(embedding_dim=128)
    model.load_state_dict(torch.load("../../PTH/lsnet_epoch_20_final.pth", map_location=device))
    model.to(device)
    model.eval()

    # -------------------------
    # 6. Embeddings
    # -------------------------
    z_ref  = embedding.get_embedding(ref_iq, model, device)
    z_test = embedding.get_embedding(test_iq, model, device)

    # -------------------------
    # 7. Hardware Identity Gate
    # -------------------------
    decision, dist = gate.hardware_identity_check(z_ref, z_test)

    print("\n========== RESULT ==========")
    print("Decision :", decision)
    print("Distance :", dist)
    print("============================")


if __name__ == "__main__":
    main()
