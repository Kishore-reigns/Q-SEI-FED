import numpy as np

print("[DEBUG] data_loader.py loaded")

def load_iq_npz(path):
    data = np.load(path, allow_pickle=True)

    print("[INFO] Keys in NPZ:", data.files)

    iq = data["iq"]

    # ---- FIX: handle complex IQ ----
    if np.iscomplexobj(iq):
        print("[INFO] IQ is complex → converting to (I,Q)")
        iq = np.stack([iq.real, iq.imag], axis=1)   # (T,2)

    elif iq.ndim == 1:
        raise ValueError("IQ is 1D but not complex. Cannot infer I/Q.")

    elif iq.shape[1] != 2:
        raise ValueError(f"Unexpected IQ shape: {iq.shape}")

    # ---- Metadata ----
    meta = {
        "snr_db": data["snr_db"].item() if "snr_db" in data else None,
        "instance_id": data["instance_id"].item() if "instance_id" in data else None,
        "manufacturer": data["manufacturer"].item() if "manufacturer" in data else None,
        "is_known": data["is_known"].item() if "is_known" in data else None,
        "sei_params": data["sei_params"].item() if "sei_params" in data else None
    }

    print(f"[INFO] IQ final shape: {iq.shape}")
    print(f"[INFO] Metadata: {meta}")

    return iq, meta
