"""
5 -> 3 trusted UAVs, 2 -> untrusted UAVs
6 -> 4 trusted UAVs, 2 -> untrusted UAVs
7 -> 5 trusted UAVs, 2 -> untrusted UAVs 
8 -> 6 trusted UAVs, 2 -> untrusted UAVs 
9 -> 7 trusted UAVs, 2 -> untrusted UAVs 
10 -> 8 trusted UAVs, 2 -> untrusted UAVs
"""

import matplotlib.pyplot as plt
OUTPUT_DIR = "plots"

rounds = [0,1, 2, 3, 4, 5, 6]

t3 = [0,-0.002, -0.0265, -0.014, -0.0005, 0.0002, 0.0028]
t4 = [0,-0.009, -0.0215, -0.0105, -0.013, -0.033, -0.0065]
t5 = [0,-0.030, -0.015, -0.005, -0.004, -0.011, -0.016]
t6 = [0,-0.015, -0.026, -0.025, -0.010, 0.005, -0.009]
t7 = [0,-0.016, -0.0165, -0.008, -0.006, 0.0005, -0.012]
t8 = [0,-0.003, -0.016, -0.004, -0.020, -0.005, -0.002]

plt.figure(figsize=(8,5))

plt.plot(rounds, t3, marker='o', label="3 Trusted, 2 Malicious")
plt.plot(rounds, t4, marker='o', label="4 Trusted, 2 Malicious")
plt.plot(rounds, t5, marker='o', label="5 Trusted, 2 Malicious")
plt.plot(rounds, t6, marker='o', label="6 Trusted, 2 Malicious")
plt.plot(rounds, t7, marker='o', label="7 Trusted, 2 Malicious")
plt.plot(rounds, t8, marker='o', label="8 Trusted, 2 Malicious")

plt.xlabel("FL Round")
plt.ylabel("Adaptive Threshold Drift (δₜ)")
plt.title("Threshold Drift Across FL Rounds\n(Trusted vs Malicious Distribution)")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/threshold_drift.png", dpi=300)

plt.show()