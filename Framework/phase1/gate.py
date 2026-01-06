import torch.nn.functional as F

THRESHOLD = 0.45   # tune later with FAR/FRR

def cosine_dist(z1, z2):
    return 1 - F.cosine_similarity(z1, z2, dim=0)

def hardware_identity_check(z_ref, z_test):
    d = cosine_dist(z_ref, z_test)

    if d < THRESHOLD:
        return "PASS (Trusted UAV)", d.item()
    else:
        return "BLOCK (Clone Detected)", d.item()
import torch.nn.functional as F

THRESHOLD = 0.45   # tune later with FAR/FRR

def cosine_dist(z1, z2):
    return 1 - F.cosine_similarity(z1, z2, dim=0)

def hardware_identity_check(z_ref, z_test):
    d = cosine_dist(z_ref, z_test)

    if d < THRESHOLD:
        return "PASS (Trusted UAV)", d.item()
    else:
        return "BLOCK (Clone Detected)", d.item()
