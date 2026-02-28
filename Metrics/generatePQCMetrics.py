import matplotlib.pyplot as plt

from generateMetrics import OUTPUT_DIR

uavs = [2,3,4,5,6,7,8,9,10]

# ===============================
# NO PQC
# ===============================
no_latency = [16.88,16.88,16.88,16.88,16.92,16.95,16.98,17.02,17.05]
no_crypto = [0,0,0,0,0,0,0,0,0]
no_bandwidth = [15.62,15.62,15.62,15.62,15.65,15.68,15.70,15.72,15.75]
no_overhead = [0.0,0.0,0.0,0.0,0.02,0.03,0.04,0.05,0.06]

# ===============================
# HYBRID PQC
# ===============================
hy_latency = [29.42,29.46,29.60,29.65,29.68,29.72,29.75,29.79,29.83]
hy_crypto = [12.49,12.53,12.66,12.68,12.71,12.74,12.77,12.80,12.83]
hy_bandwidth = [15.80,15.81,15.81,15.85,15.87,15.90,15.93,15.96,15.99]
hy_overhead = [1.19,1.2,1.2,1.2,1.3,1.32,1.38,1.40,1.43]

# ===============================
# FULL PQC
# ===============================
full_latency = [35.01,35.01,35.23,35.27,35.25,35.32,35.36,35.40,35.45]
full_crypto = [16.38,16.38,16.48,16.54,16.57,16.60,16.63,16.66,16.70]
full_bandwidth = [21.23,21.23,21.23,21.28,21.30,21.34,21.38,21.42,21.47]
full_overhead = [35.8,35.8,35.8,35.9,36.0,36.1,36.2,36.3,36.5]


# ======================================
# 1️⃣ Avg Latency Plot
# ======================================
plt.figure()
plt.plot(uavs, no_latency, label="NO PQC")
plt.plot(uavs, hy_latency, label="HYBRID")
plt.plot(uavs, full_latency, label="FULL")
plt.xlabel("Number of UAVs")
plt.ylabel("Average Latency (ms)")
plt.title("Average Latency vs UAV Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/avg_latency.png", dpi=300)
plt.show()


# ======================================
# 2️⃣ Crypto Delay Plot
# ======================================
plt.figure()
plt.plot(uavs, no_crypto, label="NO PQC")
plt.plot(uavs, hy_crypto, label="HYBRID")
plt.plot(uavs, full_crypto, label="FULL")
plt.xlabel("Number of UAVs")
plt.ylabel("Crypto Delay (ms)")
plt.title("Crypto Delay vs UAV Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/crypto_delay.png", dpi=300)
plt.show()


# ======================================
# 3️⃣ Bandwidth Plot
# ======================================
plt.figure()
plt.plot(uavs, no_bandwidth, label="NO PQC")
plt.plot(uavs, hy_bandwidth, label="HYBRID")
plt.plot(uavs, full_bandwidth, label="FULL")
plt.xlabel("Number of UAVs")
plt.ylabel("Bandwidth (KB)")
plt.title("Bandwidth vs UAV Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/bandwidth.png", dpi=300)
plt.show()


# ======================================
# 4️⃣ Overhead Plot
# ======================================
plt.figure()
plt.plot(uavs, no_overhead, label="NO PQC")
plt.plot(uavs, hy_overhead, label="HYBRID")
plt.plot(uavs, full_overhead, label="FULL")
plt.xlabel("Number of UAVs")
plt.ylabel("Overhead (%)")
plt.title("Overhead vs UAV Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/overhead.png", dpi=300)
plt.show()