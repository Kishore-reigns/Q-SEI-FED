import streamlit as st
import numpy as np
import torch
import os
import random
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import matplotlib
import io
import hashlib
import hmac
from scipy import signal
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json
import pandas as pd
from datetime import datetime, timedelta

# -----------------------
# MATPLOTLIB FIX
# -----------------------
matplotlib.use("Agg")

# -----------------------
# LSNET MODEL
# -----------------------
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)

    def forward(self, x):
        return self.dw(x)

class MCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class MCAC(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = DWConv(c)
        self.mca = MCA(c)
        self.pw1 = nn.Conv2d(c, 4*c, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4*c, c, 1)

    def forward(self, x):
        y = self.dw(x)
        y = self.mca(y)
        y = self.act(self.pw1(y))
        y = self.pw2(y)
        return x + y

class LSNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.stage1 = nn.Sequential(*[MCAC(32) for _ in range(3)])
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.stage2 = nn.Sequential(*[MCAC(64) for _ in range(4)])
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.stage3 = nn.Sequential(*[MCAC(128) for _ in range(6)])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.fc(x))

# -----------------------
# PQC EVALUATION MODULE WITH TIME-BASED METRICS
# -----------------------

@dataclass
class TimedSecurityMetrics:
    """Metrics for security protocol evaluation with timestamps"""
    protocol_type: str  # 'CLASSICAL', 'FULL_PQC', or 'HYBRID_PQC'
    handshake_time_ms: float
    encryption_time_ms: float
    decryption_time_ms: float
    bandwidth_bytes: int
    packet_size_bytes: int
    success: bool
    timestamp: float
    operation_id: str
    
class TimeBasedPQCEvaluator:
    """Evaluator with time-based metrics tracking"""
    
    # Size definitions (in bytes)
    CLASSICAL_KEY_SIZE = 32
    CLASSICAL_SIG_SIZE = 64
    DILITHIUM_KEY_SIZE = 1312
    DILITHIUM_SIG_SIZE = 2420
    KYBER_KEY_SIZE = 800
    KYBER_CIPHER_SIZE = 768
    AES_KEY_SIZE = 32
    
    def __init__(self):
        self.classical_metrics: List[TimedSecurityMetrics] = []
        self.full_pqc_metrics: List[TimedSecurityMetrics] = []
        self.hybrid_pqc_metrics: List[TimedSecurityMetrics] = []
        self.current_protocol = "HYBRID_PQC"
        self.start_time = time.time()
        self.operation_counter = 0
        
    def simulate_classical_auth(self, drone_id: str) -> TimedSecurityMetrics:
        """Simulate classical ECC/ECDSA authentication"""
        self.operation_counter += 1
        op_id = f"classical_{self.operation_counter}"
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # Simulate ECC key exchange
        ecdh_start = time.time()
        shared_key = os.urandom(self.CLASSICAL_KEY_SIZE)
        ecdh_time = (time.time() - ecdh_start) * 1000
        
        # Simulate ECDSA signature
        sign_start = time.time()
        message = f"{drone_id}_{time.time()}".encode()
        signature = hashlib.sha256(message + shared_key).digest()
        sign_time = (time.time() - sign_start) * 1000
        
        # Simulate data transmission
        encrypt_start = time.time()
        data = os.urandom(1024)
        encrypted = self._aes_encrypt(data, shared_key)
        encrypt_time = (time.time() - encrypt_start) * 1000
        
        # Simulate decryption
        decrypt_start = time.time()
        self._aes_decrypt(encrypted, shared_key)
        decrypt_time = (time.time() - decrypt_start) * 1000
        
        metrics = TimedSecurityMetrics(
            protocol_type="CLASSICAL",
            handshake_time_ms=ecdh_time + sign_time,
            encryption_time_ms=encrypt_time,
            decryption_time_ms=decrypt_time,
            bandwidth_bytes=self.CLASSICAL_KEY_SIZE + self.CLASSICAL_SIG_SIZE,
            packet_size_bytes=len(data),
            success=True,
            timestamp=relative_time,
            operation_id=op_id
        )
        
        self.classical_metrics.append(metrics)
        return metrics
    
    def simulate_full_pqc_auth(self, drone_id: str) -> TimedSecurityMetrics:
        """Simulate FULL PQC (everything uses PQC)"""
        self.operation_counter += 1
        op_id = f"full_pqc_{self.operation_counter}"
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # FULL PQC: Both handshake AND data use PQC
        kyber_start = time.time()
        session_key = os.urandom(32)
        kyber_cipher = os.urandom(self.KYBER_CIPHER_SIZE)
        kyber_time = (time.time() - kyber_start) * 1000
        
        dilithium_start = time.time()
        message = f"{drone_id}_{kyber_cipher.hex()}".encode()
        signature = hashlib.sha512(message).digest()[:self.DILITHIUM_SIG_SIZE]
        dilithium_time = (time.time() - dilithium_start) * 1000
        
        handshake_time = kyber_time + dilithium_time
        
        # FULL PQC: Data also encrypted with PQC (slower)
        encrypt_start = time.time()
        data = os.urandom(1024)
        time.sleep(0.01)  # Simulate PQC encryption overhead
        encrypted = self._pqc_encrypt(data, session_key)
        encrypt_time = (time.time() - encrypt_start) * 1000
        
        decrypt_start = time.time()
        time.sleep(0.01)  # Simulate PQC decryption overhead
        self._pqc_decrypt(encrypted, session_key)
        decrypt_time = (time.time() - decrypt_start) * 1000
        
        metrics = TimedSecurityMetrics(
            protocol_type="FULL_PQC",
            handshake_time_ms=handshake_time,
            encryption_time_ms=encrypt_time,
            decryption_time_ms=decrypt_time,
            bandwidth_bytes=self.DILITHIUM_SIG_SIZE + self.KYBER_CIPHER_SIZE + len(data),
            packet_size_bytes=len(data) + 500,
            success=True,
            timestamp=relative_time,
            operation_id=op_id
        )
        
        self.full_pqc_metrics.append(metrics)
        return metrics
    
    def simulate_hybrid_pqc_auth(self, drone_id: str) -> TimedSecurityMetrics:
        """Simulate Hybrid PQC (PQC handshake + AES data)"""
        self.operation_counter += 1
        op_id = f"hybrid_{self.operation_counter}"
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # Phase 1: Quantum-Resistant Handshake (PQC)
        kyber_start = time.time()
        session_key = os.urandom(self.AES_KEY_SIZE)
        kyber_cipher = os.urandom(self.KYBER_CIPHER_SIZE)
        kyber_time = (time.time() - kyber_start) * 1000
        
        dilithium_start = time.time()
        message = f"{drone_id}_{kyber_cipher.hex()}".encode()
        signature = hashlib.sha512(message).digest()[:self.DILITHIUM_SIG_SIZE]
        dilithium_time = (time.time() - dilithium_start) * 1000
        
        handshake_time = kyber_time + dilithium_time
        
        # Phase 2: Classical AES encryption
        encrypt_start = time.time()
        data = os.urandom(1024)
        encrypted = self._aes_encrypt(data, session_key)
        encrypt_time = (time.time() - encrypt_start) * 1000
        
        decrypt_start = time.time()
        self._aes_decrypt(encrypted, session_key)
        decrypt_time = (time.time() - decrypt_start) * 1000
        
        metrics = TimedSecurityMetrics(
            protocol_type="HYBRID_PQC",
            handshake_time_ms=handshake_time,
            encryption_time_ms=encrypt_time,
            decryption_time_ms=decrypt_time,
            bandwidth_bytes=self.DILITHIUM_SIG_SIZE + self.KYBER_CIPHER_SIZE,
            packet_size_bytes=len(data),
            success=True,
            timestamp=relative_time,
            operation_id=op_id
        )
        
        self.hybrid_pqc_metrics.append(metrics)
        return metrics
    
    def _aes_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simulate AES encryption"""
        key_hash = hashlib.sha256(key).digest()
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_hash[i % len(key_hash)])
        return bytes(encrypted)
    
    def _aes_decrypt(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Simulate AES decryption"""
        return self._aes_encrypt(encrypted_data, key)
    
    def _pqc_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simulate PQC encryption (slower)"""
        time.sleep(0.005)
        return self._aes_encrypt(data, key) + b"_PQC_OVERHEAD"
    
    def _pqc_decrypt(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Simulate PQC decryption (slower)"""
        time.sleep(0.005)
        return self._aes_decrypt(encrypted_data[:-12], key)
    
    def plot_time_based_comparison(self):
        """Create time-based comparison charts"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Combine all metrics with timestamps
        all_metrics = []
        protocol_colors = {'CLASSICAL': 'blue', 'FULL_PQC': 'red', 'HYBRID_PQC': 'green'}
        
        # Prepare time-series data
        time_series_data = {
            'CLASSICAL': {'times': [], 'handshake': [], 'encryption': [], 'total': []},
            'FULL_PQC': {'times': [], 'handshake': [], 'encryption': [], 'total': []},
            'HYBRID_PQC': {'times': [], 'handshake': [], 'encryption': [], 'total': []}
        }
        
        # Process classical metrics
        for metric in self.classical_metrics:
            time_series_data['CLASSICAL']['times'].append(metric.timestamp)
            time_series_data['CLASSICAL']['handshake'].append(metric.handshake_time_ms)
            time_series_data['CLASSICAL']['encryption'].append(metric.encryption_time_ms)
            time_series_data['CLASSICAL']['total'].append(metric.handshake_time_ms + metric.encryption_time_ms)
        
        # Process full PQC metrics
        for metric in self.full_pqc_metrics:
            time_series_data['FULL_PQC']['times'].append(metric.timestamp)
            time_series_data['FULL_PQC']['handshake'].append(metric.handshake_time_ms)
            time_series_data['FULL_PQC']['encryption'].append(metric.encryption_time_ms)
            time_series_data['FULL_PQC']['total'].append(metric.handshake_time_ms + metric.encryption_time_ms)
        
        # Process hybrid PQC metrics
        for metric in self.hybrid_pqc_metrics:
            time_series_data['HYBRID_PQC']['times'].append(metric.timestamp)
            time_series_data['HYBRID_PQC']['handshake'].append(metric.handshake_time_ms)
            time_series_data['HYBRID_PQC']['encryption'].append(metric.encryption_time_ms)
            time_series_data['HYBRID_PQC']['total'].append(metric.handshake_time_ms + metric.encryption_time_ms)
        
        # Plot 1: Handshake Time Over Time
        ax1 = axes[0, 0]
        for protocol, data in time_series_data.items():
            if data['times']:
                # Sort by time
                sorted_indices = np.argsort(data['times'])
                times = np.array(data['times'])[sorted_indices]
                handshake = np.array(data['handshake'])[sorted_indices]
                ax1.plot(times, handshake, 'o-', color=protocol_colors[protocol], 
                        label=protocol, markersize=4, alpha=0.7)
        ax1.set_xlabel('Time (seconds from start)')
        ax1.set_ylabel('Handshake Time (ms)')
        ax1.set_title('Handshake Time Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Data Encryption Time Over Time
        ax2 = axes[0, 1]
        for protocol, data in time_series_data.items():
            if data['times'] and data['encryption']:
                sorted_indices = np.argsort(data['times'])
                times = np.array(data['times'])[sorted_indices]
                encryption = np.array(data['encryption'])[sorted_indices]
                ax2.plot(times, encryption, 's-', color=protocol_colors[protocol], 
                        label=f"{protocol} Data", markersize=4, alpha=0.7)
        ax2.set_xlabel('Time (seconds from start)')
        ax2.set_ylabel('Data Encryption Time (ms)')
        ax2.set_title('Data Encryption Time Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total Operation Time Over Time
        ax3 = axes[1, 0]
        for protocol, data in time_series_data.items():
            if data['times'] and data['total']:
                sorted_indices = np.argsort(data['times'])
                times = np.array(data['times'])[sorted_indices]
                total = np.array(data['total'])[sorted_indices]
                ax3.plot(times, total, '^-', color=protocol_colors[protocol], 
                        label=protocol, linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time (seconds from start)')
        ax3.set_ylabel('Total Time (ms)')
        ax3.set_title('Total Operation Time (Handshake + Data) Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Moving Average Comparison
        ax4 = axes[1, 1]
        window_size = 3
        for protocol, data in time_series_data.items():
            if data['times'] and data['total']:
                sorted_indices = np.argsort(data['times'])
                times = np.array(data['times'])[sorted_indices]
                total = np.array(data['total'])[sorted_indices]
                
                # Calculate moving average
                if len(total) >= window_size:
                    moving_avg = np.convolve(total, np.ones(window_size)/window_size, mode='valid')
                    moving_times = times[window_size-1:]
                    ax4.plot(moving_times, moving_avg, '--', color=protocol_colors[protocol], 
                            label=f"{protocol} (MA)", linewidth=2, alpha=0.8)
        ax4.set_xlabel('Time (seconds from start)')
        ax4.set_ylabel('Moving Average (ms)')
        ax4.set_title(f'{window_size}-Point Moving Average of Total Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative Time Over Time
        ax5 = axes[2, 0]
        for protocol, data in time_series_data.items():
            if data['times'] and data['total']:
                sorted_indices = np.argsort(data['times'])
                times = np.array(data['times'])[sorted_indices]
                total = np.array(data['total'])[sorted_indices]
                cumulative = np.cumsum(total)
                ax5.plot(times, cumulative, '-', color=protocol_colors[protocol], 
                        label=protocol, linewidth=2)
        ax5.set_xlabel('Time (seconds from start)')
        ax5.set_ylabel('Cumulative Time (ms)')
        ax5.set_title('Cumulative Operation Time Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Hybrid PQC Efficiency Gain Over Time
        ax6 = axes[2, 1]
        if (self.full_pqc_metrics and self.hybrid_pqc_metrics and 
            len(self.full_pqc_metrics) >= 2 and len(self.hybrid_pqc_metrics) >= 2):
            
            # Get matching timestamps
            min_len = min(len(self.full_pqc_metrics), len(self.hybrid_pqc_metrics))
            full_times = [m.timestamp for m in self.full_pqc_metrics[:min_len]]
            hybrid_times = [m.timestamp for m in self.hybrid_pqc_metrics[:min_len]]
            full_encrypt = [m.encryption_time_ms for m in self.full_pqc_metrics[:min_len]]
            hybrid_encrypt = [m.encryption_time_ms for m in self.hybrid_pqc_metrics[:min_len]]
            
            # Calculate improvement percentage over time
            improvements = []
            valid_times = []
            for i in range(min_len):
                if full_encrypt[i] > 0:
                    improvement = ((full_encrypt[i] - hybrid_encrypt[i]) / full_encrypt[i]) * 100
                    improvements.append(improvement)
                    valid_times.append((full_times[i] + hybrid_times[i]) / 2)
            
            if improvements:
                ax6.plot(valid_times, improvements, 'o-', color='purple', 
                        linewidth=2, markersize=5)
                ax6.axhline(y=np.mean(improvements), color='red', linestyle='--', 
                          label=f'Avg: {np.mean(improvements):.1f}%')
                ax6.set_xlabel('Time (seconds from start)')
                ax6.set_ylabel('Improvement %')
                ax6.set_title('Hybrid vs Full PQC: Data Encryption Improvement Over Time')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_time_based_stats(self):
        """Get time-based statistics"""
        stats = []
        
        for protocol, metrics_list in [('CLASSICAL', self.classical_metrics),
                                      ('FULL_PQC', self.full_pqc_metrics),
                                      ('HYBRID_PQC', self.hybrid_pqc_metrics)]:
            if metrics_list:
                times = [m.timestamp for m in metrics_list]
                handshakes = [m.handshake_time_ms for m in metrics_list]
                encryptions = [m.encryption_time_ms for m in metrics_list]
                totals = [h + e for h, e in zip(handshakes, encryptions)]
                
                stats.append({
                    'Protocol': protocol,
                    'First Op Time': f"{min(times):.1f}s",
                    'Last Op Time': f"{max(times):.1f}s",
                    'Time Span': f"{max(times)-min(times):.1f}s",
                    'Avg Handshake': f"{np.mean(handshakes):.1f}ms",
                    'Avg Encryption': f"{np.mean(encryptions):.1f}ms",
                    'Avg Total': f"{np.mean(totals):.1f}ms",
                    'Operations': len(metrics_list),
                    'Ops per Minute': f"{len(metrics_list)/(max(times)-min(times)+0.1)*60:.1f}"
                })
        
        return pd.DataFrame(stats)
    
    def get_hybrid_efficiency_over_time(self):
        """Calculate hybrid efficiency improvements over time"""
        if not self.full_pqc_metrics or not self.hybrid_pqc_metrics:
            return None
        
        min_len = min(len(self.full_pqc_metrics), len(self.hybrid_pqc_metrics))
        efficiency_data = []
        
        for i in range(min_len):
            full = self.full_pqc_metrics[i]
            hybrid = self.hybrid_pqc_metrics[i]
            
            # Calculate improvements
            handshake_improvement = ((full.handshake_time_ms - hybrid.handshake_time_ms) / 
                                    full.handshake_time_ms * 100) if full.handshake_time_ms > 0 else 0
            encryption_improvement = ((full.encryption_time_ms - hybrid.encryption_time_ms) / 
                                     full.encryption_time_ms * 100) if full.encryption_time_ms > 0 else 0
            total_improvement = (((full.handshake_time_ms + full.encryption_time_ms) - 
                                (hybrid.handshake_time_ms + hybrid.encryption_time_ms)) / 
                               (full.handshake_time_ms + full.encryption_time_ms) * 100)
            
            efficiency_data.append({
                'Time': full.timestamp,
                'Handshake Improvement %': handshake_improvement,
                'Data Encryption Improvement %': encryption_improvement,
                'Total Improvement %': total_improvement,
                'Full PQC Total (ms)': full.handshake_time_ms + full.encryption_time_ms,
                'Hybrid PQC Total (ms)': hybrid.handshake_time_ms + hybrid.encryption_time_ms
            })
        
        return pd.DataFrame(efficiency_data)

# -----------------------
# BASE STATION
# -----------------------

class BaseStation:
    """Ground Control System with Time-based Evaluation"""
    
    def __init__(self):
        self.evaluator = TimeBasedPQCEvaluator()
        self.active_sessions = {}
        self.protocol_mode = "HYBRID_PQC"
        self.metrics_history = []
        
    def authenticate_drone(self, drone, force_protocol: str = None):
        """Authenticate drone with selected protocol"""
        protocol = force_protocol or self.protocol_mode
        
        if protocol == "CLASSICAL":
            metrics = self.evaluator.simulate_classical_auth(drone.id)
            auth_type = "Classical ECC/ECDSA"
        elif protocol == "FULL_PQC":
            metrics = self.evaluator.simulate_full_pqc_auth(drone.id)
            auth_type = "Full PQC (Dilithium+Kyber for everything)"
        else:  # HYBRID_PQC
            metrics = self.evaluator.simulate_hybrid_pqc_auth(drone.id)
            auth_type = "Hybrid PQC (PQC handshake + AES data)"
        
        if metrics.success:
            self.active_sessions[drone.id] = {
                'session_key': os.urandom(32),
                'authenticated_at': time.time(),
                'protocol': protocol,
                'metrics': metrics
            }
            drone.pqc_authenticated = True
            drone.protocol_used = protocol
            drone.auth_metrics = metrics
            drone.auth_timestamp = metrics.timestamp
            
            self.metrics_history.append({
                'drone_id': drone.id,
                'protocol': protocol,
                'handshake_time': metrics.handshake_time_ms,
                'encryption_time': metrics.encryption_time_ms,
                'bandwidth': metrics.bandwidth_bytes,
                'timestamp': metrics.timestamp
            })
            
            return True, metrics, auth_type
        else:
            return False, metrics, auth_type
    
    def switch_protocol(self, new_protocol: str):
        """Switch between protocols"""
        if new_protocol in ["CLASSICAL", "FULL_PQC", "HYBRID_PQC"]:
            self.protocol_mode = new_protocol
            return True
        return False

# -----------------------
# CONFIG
# -----------------------
DATASET_ROOT = r"F:\K DRIVE\MIT_Learnings\Sem8\dataset\code\sei_dataset4"

if not os.path.exists(DATASET_ROOT):
    st.error(f"Dataset path not found: {DATASET_ROOT}")
    st.stop()

MAX_DRONES = 6
INITIAL_DRONES = 3
WORLD_SIZE = 1.0
COMM_RANGE = 0.25
THRESHOLD = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colors for 3 protocols
COLORS = {
    'trusted': '#00FF00',
    'malicious': '#FF0000',
    'communication': '#4169E1',
    'isolated': '#FFA500',
    'classical_auth': '#9370DB',
    'full_pqc_auth': '#FF6B6B',
    'hybrid_pqc_auth': '#00CED1',
    'enemy': '#FF4500',
    'unknown': '#FFD700',
}

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    try:
        model_paths = [
            "../../PTH/lsnet_epoch_20_final.pth",
            "lsnet_epoch_20_final.pth",
            "./PTH/lsnet_epoch_20_final.pth",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            import glob
            pth_files = glob.glob("**/*.pth", recursive=True)
            if pth_files:
                model_path = pth_files[0]
            else:
                model = LSNet()
                model.to(DEVICE)
                model.eval()
                return model
        
        model = LSNet()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        model = LSNet()
        model.to(DEVICE)
        model.eval()
        return model

model = load_model()

# -----------------------
# DRONE CLASS WITH RF DATA
# -----------------------

class Drone:
    def __init__(self, drone_id, label, model_name, npz_path, model_instance):
        self.id = drone_id
        self.label = label
        self.model_name = model_name
        self.npz_path = npz_path
        self.model = model_instance
        
        # Protocol attributes
        self.pqc_authenticated = False
        self.protocol_used = None
        self.auth_metrics = None
        self.auth_timestamp = None
        self.session_key = None
        
        # RF Data attributes
        self.iq_data = None
        self.spectrogram_data = None
        self.last_verification_result = None
        self.last_verification_distance = None
        
        # Load RF data immediately
        self.load_rf_data()
        
        # Physical attributes
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            self.pos = np.array([random.uniform(0.1, WORLD_SIZE-0.1), WORLD_SIZE-0.1])
        elif side == 'bottom':
            self.pos = np.array([random.uniform(0.1, WORLD_SIZE-0.1), 0.1])
        elif side == 'left':
            self.pos = np.array([0.1, random.uniform(0.1, WORLD_SIZE-0.1)])
        else:
            self.pos = np.array([WORLD_SIZE-0.1, random.uniform(0.1, WORLD_SIZE-0.1)])
        
        center = np.array([WORLD_SIZE/2, WORLD_SIZE/2])
        self.direction = center - self.pos
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        self.direction += np.random.randn(2) * 0.2
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        
        # Security attributes
        self.trusted = True if 'enemy' not in model_name.lower() else False
        self.isolated = False
        self.communication_history = []
        self.speed = 0.008 if 'enemy' in model_name.lower() else 0.005
    
    def load_rf_data(self):
        """Load IQ data and spectrogram from NPZ file"""
        try:
            data = np.load(self.npz_path, allow_pickle=True)
            
            # Try different possible keys for IQ data
            iq_keys = ['iq_data', 'iq', 'signal', 'samples', 'data']
            for key in iq_keys:
                if key in data:
                    self.iq_data = data[key]
                    # If data is too long, take first 2048 samples for visualization
                    if hasattr(self.iq_data, '__len__') and len(self.iq_data) > 2048:
                        self.iq_data = self.iq_data[:2048]
                    break
            
            # Try different possible keys for spectrogram
            spec_keys = ['spectrogram', 'spec', 'spectro', 'stft', 'spectrogram_data']
            for key in spec_keys:
                if key in data:
                    self.spectrogram_data = data[key]
                    break
            
            # If no spectrogram but have IQ data, create one
            if self.spectrogram_data is None and self.iq_data is not None:
                iq_samples = self.iq_data
                if isinstance(iq_samples, np.ndarray):
                    # Ensure we have complex data
                    if iq_samples.dtype == np.complex128 or iq_samples.dtype == np.complex64:
                        f, t, Sxx = signal.spectrogram(iq_samples, fs=1000, nperseg=64, noverlap=32)
                        self.spectrogram_data = Sxx
                    else:
                        # If real data, create complex signal
                        complex_iq = iq_samples.astype(np.complex128)
                        if len(iq_samples.shape) > 1:
                            complex_iq = iq_samples[:, 0] + 1j * iq_samples[:, 1]
                        f, t, Sxx = signal.spectrogram(complex_iq, fs=1000, nperseg=64, noverlap=32)
                        self.spectrogram_data = Sxx
            
            # Create dummy data if still None
            if self.iq_data is None:
                # Create realistic IQ data (complex signal)
                t = np.linspace(0, 1, 1024)
                carrier_freq = 100  # Hz
                self.iq_data = np.exp(1j * 2 * np.pi * carrier_freq * t)
                self.iq_data += 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
            
            if self.spectrogram_data is None:
                # Create dummy spectrogram
                t = np.linspace(0, 1, 100)
                f = np.linspace(0, 500, 64)
                self.spectrogram_data = np.random.rand(64, 100)
                
        except Exception as e:
            # Create realistic dummy data
            t = np.linspace(0, 1, 1024)
            carrier_freq = 100 + random.randint(-20, 20)
            self.iq_data = np.exp(1j * 2 * np.pi * carrier_freq * t)
            self.iq_data += 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
            
            t = np.linspace(0, 1, 100)
            f = np.linspace(0, 500, 64)
            self.spectrogram_data = np.random.rand(64, 100)
    
    def move(self):
        """Move drone in swarm"""
        if self.isolated:
            self.direction += np.random.randn(2) * 0.15
            speed = self.speed * 0.5
        else:
            self.direction += np.random.randn(2) * 0.08
            speed = self.speed
        
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        
        self.pos += self.direction * speed
        
        # Boundary checking
        bounce_randomness = 0.1
        if self.pos[0] < 0.02:
            self.pos[0] = 0.02
            self.direction[0] = abs(self.direction[0]) + random.uniform(0, bounce_randomness)
        elif self.pos[0] > WORLD_SIZE - 0.02:
            self.pos[0] = WORLD_SIZE - 0.02
            self.direction[0] = -abs(self.direction[0]) - random.uniform(0, bounce_randomness)
        
        if self.pos[1] < 0.02:
            self.pos[1] = 0.02
            self.direction[1] = abs(self.direction[1]) + random.uniform(0, bounce_randomness)
        elif self.pos[1] > WORLD_SIZE - 0.02:
            self.pos[1] = WORLD_SIZE - 0.02
            self.direction[1] = -abs(self.direction[1]) - random.uniform(0, bounce_randomness)
        
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
    
    def verify_other_drone(self, other_drone):
        """Verify another drone using LSNet"""
        try:
            # For simulation, we'll use a simplified verification
            if self.model_name == other_drone.model_name:
                # Same model - low distance (trusted)
                distance = random.uniform(0.1, 0.4)
            else:
                # Different model - higher distance
                if 'enemy' in other_drone.model_name.lower():
                    distance = random.uniform(1.5, 2.5)
                else:
                    distance = random.uniform(0.8, 1.2)
            
            is_trusted = distance < THRESHOLD
            
            self.last_verification_result = is_trusted
            self.last_verification_distance = distance
            
            return is_trusted, distance
            
        except Exception as e:
            distance = random.uniform(0.5, 2.0)
            is_trusted = distance < THRESHOLD
            return is_trusted, distance

# -----------------------
# RF DATA VISUALIZATION FUNCTIONS
# -----------------------

def plot_comprehensive_iq_analysis(iq_data, title):
    """Create comprehensive IQ data visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    if iq_data is None:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No IQ Data Available', 
                   ha='center', va='center', fontsize=12)
        return fig
    
    # Ensure we have data to plot
    samples_to_plot = min(512, len(iq_data))
    
    # Plot 1: I Component Time Series
    axes[0, 0].plot(np.real(iq_data)[:samples_to_plot], color='cyan', linewidth=1.5)
    axes[0, 0].set_title(f'{title} - In-phase Component (I)', fontsize=11)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q Component Time Series
    axes[0, 1].plot(np.imag(iq_data)[:samples_to_plot], color='magenta', linewidth=1.5)
    axes[0, 1].set_title(f'{title} - Quadrature Component (Q)', fontsize=11)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: IQ Constellation Diagram
    const_samples = min(256, len(iq_data))
    scatter = axes[0, 2].scatter(np.real(iq_data)[:const_samples], 
                                 np.imag(iq_data)[:const_samples], 
                                 c=range(const_samples), cmap='viridis',
                                 alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    axes[0, 2].set_title(f'{title} - Constellation Diagram', fontsize=11)
    axes[0, 2].set_xlabel('I (In-phase)')
    axes[0, 2].set_ylabel('Q (Quadrature)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[0, 2], label='Sample Index')
    
    # Plot 4: Magnitude Time Series
    magnitude = np.abs(iq_data)[:samples_to_plot]
    axes[1, 0].plot(magnitude, color='orange', linewidth=1.5)
    axes[1, 0].set_title(f'{title} - Signal Magnitude', fontsize=11)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Phase Time Series
    phase = np.angle(iq_data)[:samples_to_plot]
    axes[1, 1].plot(phase, color='green', linewidth=1.5)
    axes[1, 1].set_title(f'{title} - Signal Phase', fontsize=11)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Histogram of I and Q components
    axes[1, 2].hist(np.real(iq_data)[:samples_to_plot], bins=30, alpha=0.5, 
                   color='cyan', label='I Component', density=True)
    axes[1, 2].hist(np.imag(iq_data)[:samples_to_plot], bins=30, alpha=0.5, 
                   color='magenta', label='Q Component', density=True)
    axes[1, 2].set_title(f'{title} - IQ Distribution', fontsize=11)
    axes[1, 2].set_xlabel('Amplitude')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_enhanced_spectrogram(spectrogram_data, title):
    """Create enhanced spectrogram visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if spectrogram_data is None or spectrogram_data.ndim != 2:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'Invalid Spectrogram Data', 
                   ha='center', va='center', fontsize=12)
        return fig
    
    # Ensure reasonable size for visualization
    if spectrogram_data.shape[0] > 256:
        spectrogram_data = spectrogram_data[:256, :]
    if spectrogram_data.shape[1] > 256:
        spectrogram_data = spectrogram_data[:, :256]
    
    # Plot 1: Log-scaled Spectrogram (dB)
    im1 = axes[0, 0].imshow(10 * np.log10(spectrogram_data + 1e-10), 
                           aspect='auto', origin='lower', 
                           cmap='hot', interpolation='bilinear')
    axes[0, 0].set_title(f'{title} - Power Spectrogram (dB)', fontsize=11)
    axes[0, 0].set_xlabel('Time Samples')
    axes[0, 0].set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=axes[0, 0], label='Power (dB)')
    
    # Plot 2: Magnitude Spectrogram
    im2 = axes[0, 1].imshow(np.abs(spectrogram_data), aspect='auto', origin='lower',
                           cmap='viridis', interpolation='bilinear')
    axes[0, 1].set_title(f'{title} - Magnitude Spectrogram', fontsize=11)
    axes[0, 1].set_xlabel('Time Samples')
    axes[0, 1].set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=axes[0, 1], label='Magnitude')
    
    # Plot 3: Frequency Profile (average over time)
    freq_profile = np.mean(spectrogram_data, axis=1)
    axes[1, 0].plot(freq_profile, color='blue', linewidth=2)
    axes[1, 0].set_title(f'{title} - Frequency Profile', fontsize=11)
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Average Power')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Time Profile (average over frequency)
    time_profile = np.mean(spectrogram_data, axis=0)
    axes[1, 1].plot(time_profile, color='red', linewidth=2)
    axes[1, 1].set_title(f'{title} - Time Profile', fontsize=11)
    axes[1, 1].set_xlabel('Time Sample')
    axes[1, 1].set_ylabel('Average Power')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_rf_fingerprint_comparison(drone1, drone2):
    """Compare RF fingerprints of two drones"""
    if drone1.iq_data is None or drone2.iq_data is None:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Take first 256 samples for comparison
    samples = 256
    iq1 = drone1.iq_data[:samples] if len(drone1.iq_data) >= samples else drone1.iq_data
    iq2 = drone2.iq_data[:samples] if len(drone2.iq_data) >= samples else drone2.iq_data
    
    # Plot 1: I Component Comparison
    axes[0, 0].plot(np.real(iq1), 'b-', label=f'{drone1.label} (I)', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(np.real(iq2), 'r--', label=f'{drone2.label} (I)', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('In-phase Component Comparison', fontsize=11)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q Component Comparison
    axes[0, 1].plot(np.imag(iq1), 'b-', label=f'{drone1.label} (Q)', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(np.imag(iq2), 'r--', label=f'{drone2.label} (Q)', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_title('Quadrature Component Comparison', fontsize=11)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Constellation Comparison
    axes[1, 0].scatter(np.real(iq1), np.imag(iq1), s=20, alpha=0.6, 
                      c='blue', label=drone1.label, edgecolors='black', linewidths=0.5)
    axes[1, 0].scatter(np.real(iq2), np.imag(iq2), s=20, alpha=0.6, 
                      c='red', label=drone2.label, edgecolors='black', linewidths=0.5)
    axes[1, 0].set_title('Constellation Diagram Comparison', fontsize=11)
    axes[1, 0].set_xlabel('I (In-phase)')
    axes[1, 0].set_ylabel('Q (Quadrature)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # Plot 4: Magnitude Comparison
    axes[1, 1].plot(np.abs(iq1), 'b-', label=f'{drone1.label}', alpha=0.7, linewidth=1.5)
    axes[1, 1].plot(np.abs(iq2), 'r--', label=f'{drone2.label}', alpha=0.7, linewidth=1.5)
    axes[1, 1].set_title('Signal Magnitude Comparison', fontsize=11)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'RF Fingerprint Comparison: {drone1.label} vs {drone2.label}', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# -----------------------
# UTILITY FUNCTIONS
# -----------------------

def pick_random_drone():
    """Pick random drone from dataset"""
    try:
        models = [d for d in os.listdir(DATASET_ROOT) 
                 if os.path.isdir(os.path.join(DATASET_ROOT, d))]
        
        if not models:
            return "simulated", "dummy_path.npz"
        
        model_weights = []
        for m in models:
            if 'enemy' in m.lower():
                model_weights.append(0.3)
            elif 'unknown' in m.lower():
                model_weights.append(0.2)
            else:
                model_weights.append(0.1)
        
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w/total_weight for w in model_weights]
            model_name = random.choices(models, weights=model_weights)[0]
        else:
            model_name = random.choice(models)
        
        model_path = os.path.join(DATASET_ROOT, model_name)
        npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
        
        if not npz_files:
            npz_path = f"dummy_{model_name}.npz"
        else:
            npz_file = random.choice(npz_files)
            npz_path = os.path.join(model_path, npz_file)
        
        return model_name, npz_path
        
    except Exception as e:
        return "simulated", "dummy_path.npz"

def create_initial_drones():
    """Create initial drones"""
    drones = []
    
    for i in range(INITIAL_DRONES):
        model_name, npz_path = pick_random_drone()
        
        drone = Drone(
            drone_id=f"D{i+1}",
            label=f"{model_name}_{i+1}",
            model_name=model_name,
            npz_path=npz_path,
            model_instance=model
        )
        drones.append(drone)
    
    return drones

# -----------------------
# SESSION STATE INITIALIZATION
# -----------------------

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        st.session_state.simulation_running = True
        st.session_state.pause_time = 0
        st.session_state.last_update = time.time()
        st.session_state.last_auth_cycle = 0
        st.session_state.simulation_start_time = time.time()
        
        st.session_state.drones = create_initial_drones()
        st.session_state.selected_drone = None
        st.session_state.comparison_drone = None
        
        st.session_state.logs = []
        st.session_state.max_logs = 100
        
        st.session_state.communications = []
        st.session_state.malicious_drones = set()
        
        # Initialize Base Station
        st.session_state.base_station = BaseStation()
        
        # Add initial logs
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.logs.append(f"[{timestamp}] 🚀 Q-SEI-Fed Phase 2: Time-based PQC Evaluation Started")
        st.session_state.logs.append(f"[{timestamp}] 📊 All metrics now show time evolution on X-axis")
        st.session_state.logs.append(f"[{timestamp}] 📡 Click any drone to view its RF fingerprints")
        st.session_state.logs.append(f"[{timestamp}] 🛸 Initialized with {INITIAL_DRONES} drones")

initialize_session_state()

# -----------------------
# LOGGING FUNCTIONS
# -----------------------

def add_log(message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    
    if level == "WARNING":
        emoji = "⚠️"
    elif level == "ERROR":
        emoji = "❌"
    elif "PQC" in message or "QUANTUM" in message:
        emoji = "🔐"
    elif "CLASSICAL" in message:
        emoji = "🔑"
    elif "FULL PQC" in message:
        emoji = "⚛️"
    elif "MALICIOUS" in message:
        emoji = "🚨"
    elif "COMM" in message:
        emoji = "📡"
    elif "ADDED" in message:
        emoji = "🆕"
    elif "RF" in message or "SPECTROGRAM" in message:
        emoji = "📶"
    elif "TIME" in message or "METRIC" in message:
        emoji = "⏱️"
    else:
        emoji = "ℹ️"
    
    log_entry = f"[{timestamp}] {emoji} {message}"
    st.session_state.logs.append(log_entry)
    
    if len(st.session_state.logs) > st.session_state.max_logs:
        st.session_state.logs = st.session_state.logs[-st.session_state.max_logs:]

# -----------------------
# SIMULATION FUNCTIONS
# -----------------------

def update_simulation():
    if not st.session_state.simulation_running:
        return
    
    current_time = time.time()
    
    # Update drones every 0.1s
    if current_time - st.session_state.last_update < 0.1:
        return
    
    st.session_state.last_update = current_time
    
    # -----------------------
    # AUTHENTICATION CYCLE (Every 3 seconds)
    # -----------------------
    if current_time - st.session_state.last_auth_cycle > 3.0:
        for drone in st.session_state.drones:
            if drone.trusted and not drone.pqc_authenticated and not drone.isolated:
                success, metrics, auth_type = st.session_state.base_station.authenticate_drone(drone)
                
                if success:
                    add_log(f"AUTH: {drone.label} with {auth_type} "
                           f"(Handshake: {metrics.handshake_time_ms:.1f}ms, "
                           f"Data: {metrics.encryption_time_ms:.1f}ms)")
                else:
                    drone.isolated = True
                    add_log(f"AUTH FAILED: {drone.label}", "WARNING")
        
        st.session_state.last_auth_cycle = current_time
    
    # -----------------------
    # DRONE MOVEMENT
    # -----------------------
    for drone in st.session_state.drones:
        drone.move()
    
    # -----------------------
    # DRONE-TO-DRONE COMMUNICATION
    # -----------------------
    st.session_state.communications = []
    
    for i in range(len(st.session_state.drones)):
        for j in range(i + 1, len(st.session_state.drones)):
            drone_a = st.session_state.drones[i]
            drone_b = st.session_state.drones[j]
            
            if drone_a.isolated or drone_b.isolated:
                continue
            
            distance = euclidean(drone_a.pos, drone_b.pos)
            
            if distance < COMM_RANGE:
                comm_id = f"{drone_a.id}-{drone_b.id}"
                st.session_state.communications.append((drone_a, drone_b, distance, comm_id))
                
                if drone_a.trusted and drone_b.trusted:
                    trusted_a_to_b, dist_a_to_b = drone_a.verify_other_drone(drone_b)
                    trusted_b_to_a, dist_b_to_a = drone_b.verify_other_drone(drone_a)
                    
                    max_distance = max(dist_a_to_b, dist_b_to_a)
                    is_trusted = max_distance < THRESHOLD
                    
                    if is_trusted:
                        add_log(f"COMM: {drone_a.label} ↔ {drone_b.label} | "
                               f"Dist: {distance:.3f} | Score: {max_distance:.3f}")
                    
                    if not is_trusted:
                        if dist_a_to_b >= dist_b_to_a:
                            malicious_drone = drone_b
                            detector_drone = drone_a
                        else:
                            malicious_drone = drone_a
                            detector_drone = drone_b
                        
                        malicious_drone.trusted = False
                        malicious_drone.isolated = True
                        st.session_state.malicious_drones.add(malicious_drone.id)
                        
                        add_log(f"MALICIOUS DETECTED: {malicious_drone.label} by {detector_drone.label}", "WARNING")

# -----------------------
# VISUALIZATION FUNCTIONS
# -----------------------

def create_swarm_visualization():
    """Create swarm visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    margin = 0.05
    ax.set_xlim(-margin, WORLD_SIZE + margin)
    ax.set_ylim(-margin, WORLD_SIZE + margin)
    
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    
    # Draw communication ranges
    for drone in st.session_state.drones:
        if not drone.isolated:
            circle = plt.Circle(drone.pos, COMM_RANGE, 
                              color='blue', alpha=0.03, fill=True, 
                              linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
    
    # Draw communication lines
    for comm in st.session_state.communications:
        drone_a, drone_b, distance, comm_id = comm
        
        if drone_a.trusted and drone_b.trusted:
            line_color = 'cyan'
            line_alpha = 0.7
            line_style = '-'
        else:
            line_color = 'red'
            line_alpha = 0.4
            line_style = '--'
        
        ax.annotate("", xy=drone_b.pos, xytext=drone_a.pos,
                   arrowprops=dict(arrowstyle="->", 
                                  color=line_color,
                                  lw=1.5,
                                  alpha=line_alpha,
                                  linestyle=line_style))
    
    # Draw drones
    for drone in st.session_state.drones:
        # Determine color based on authentication protocol
        if drone.isolated:
            face_color = COLORS['isolated']
            edge_color = 'red'
            size = 350
            marker = 'X'
            alpha = 0.8
        elif not drone.trusted:
            face_color = COLORS['malicious']
            edge_color = 'darkred'
            size = 300
            marker = 'o'
            alpha = 1.0
        elif drone.pqc_authenticated:
            if drone.protocol_used == "CLASSICAL":
                face_color = COLORS['classical_auth']
                edge_color = 'darkviolet'
                marker = 's'
            elif drone.protocol_used == "FULL_PQC":
                face_color = COLORS['full_pqc_auth']
                edge_color = 'darkred'
                marker = '^'
            else:  # HYBRID_PQC
                face_color = COLORS['hybrid_pqc_auth']
                edge_color = 'darkcyan'
                marker = 'D'
            size = 280
            alpha = 1.0
        elif 'enemy' in drone.model_name.lower():
            face_color = COLORS['enemy']
            edge_color = 'darkorange'
            size = 280
            marker = 'o'
            alpha = 0.9
        else:
            face_color = COLORS['trusted']
            edge_color = 'darkgreen'
            size = 250
            marker = 'o'
            alpha = 1.0
        
        # Draw drone
        ax.scatter(drone.pos[0], drone.pos[1], 
                  s=size, c=face_color, marker=marker,
                  edgecolors=edge_color, linewidths=2, alpha=alpha, zorder=10)
        
        # Add label
        protocol_emoji = ""
        if drone.pqc_authenticated:
            if drone.protocol_used == "CLASSICAL":
                protocol_emoji = "🔑"
            elif drone.protocol_used == "FULL_PQC":
                protocol_emoji = "⚛️"
            else:
                protocol_emoji = "🔐"
        
        label_text = f"{drone.label}{protocol_emoji}"
        
        ax.text(drone.pos[0], drone.pos[1] - 0.035, 
               label_text, fontsize=9, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", 
                        alpha=0.8, edgecolor='white', linewidth=0.5),
               color='white', fontweight='bold', zorder=11)
    
    # Customize plot
    ax.set_aspect('equal')
    simulation_time = time.time() - st.session_state.simulation_start_time
    ax.set_title(f'Swarm Simulation | Elapsed Time: {simulation_time:.1f}s', 
                color='white', fontsize=16, pad=20, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=COLORS['trusted'], edgecolor='darkgreen', label='Trusted'),
        Patch(facecolor=COLORS['classical_auth'], edgecolor='darkviolet', label='Classical'),
        Patch(facecolor=COLORS['full_pqc_auth'], edgecolor='darkred', label='Full PQC'),
        Patch(facecolor=COLORS['hybrid_pqc_auth'], edgecolor='darkcyan', label='Hybrid PQC'),
        Patch(facecolor=COLORS['malicious'], edgecolor='darkred', label='Malicious'),
        Patch(facecolor=COLORS['isolated'], edgecolor='red', label='Isolated'),
        Line2D([0], [0], color='cyan', lw=2, label='Secure Comm'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Blocked')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left',
             facecolor='black', edgecolor='white',
             labelcolor='white', fontsize=8)
    
    # Add info panel
    total_drones = len(st.session_state.drones)
    authenticated = sum(1 for d in st.session_state.drones if d.pqc_authenticated)
    
    info_text = (f'Drones: {total_drones}/{MAX_DRONES} | '
                f'Authenticated: {authenticated} | '
                f'Simulation Time: {simulation_time:.1f}s')
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, color='white', ha='left', va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    plt.tight_layout()
    return fig

# -----------------------
# STREAMLIT UI
# -----------------------

st.set_page_config(
    page_title="Q-SEI-Fed Phase 2: Time-based PQC Evaluation",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .protocol-card {
        background: rgba(15, 52, 96, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .protocol-card:hover {
        transform: translateX(5px);
        background: rgba(15, 52, 96, 0.2);
    }
    
    .classical-card {
        border-left-color: #9370DB !important;
        background: rgba(147, 112, 219, 0.1) !important;
    }
    
    .full-pqc-card {
        border-left-color: #FF6B6B !important;
        background: rgba(255, 107, 107, 0.1) !important;
    }
    
    .hybrid-pqc-card {
        border-left-color: #00CED1 !important;
        background: rgba(0, 206, 209, 0.1) !important;
    }
    
    .rf-data-panel {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #00CED1;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(15, 52, 96, 0.3);
    }
    
    /* Custom scrollbar */
    div[data-testid="stVerticalBlock"] > div {
        max-height: 400px;
        overflow-y: auto;
    }
    
    div[data-testid="stVerticalBlock"] > div::-webkit-scrollbar {
        width: 8px;
    }
    
    div[data-testid="stVerticalBlock"] > div::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 4px;
    }
    
    div[data-testid="stVerticalBlock"] > div::-webkit-scrollbar-thumb {
        background: #533483;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⏱️ Q-SEI-Fed Phase 2: Time-based PQC Evaluation</h1>
    <h3>All metrics now show time evolution on X-axis | Click drones to view RF fingerprints</h3>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1.2, 2])

# =======================
# LEFT PANEL - CONTROLS & DRONES
# =======================
with col1:
    st.markdown("### 🎮 Controls & Drones")
    
    # Add Drone Button
    st.markdown("#### 🛸 Add New Drone")
    if len(st.session_state.drones) < MAX_DRONES:
        if st.button("➕ Add New Drone to Swarm", use_container_width=True, key="add_drone"):
            model_name, npz_path = pick_random_drone()
            new_id = f"D{len(st.session_state.drones) + 1}"
            new_drone = Drone(
                drone_id=new_id,
                label=f"{model_name}_{len(st.session_state.drones) + 1}",
                model_name=model_name,
                npz_path=npz_path,
                model_instance=model
            )
            st.session_state.drones.append(new_drone)
            add_log(f"ADDED: New drone {new_drone.label} with RF data")
            st.rerun()
    else:
        st.warning(f"⚠️ Maximum {MAX_DRONES} drones reached")
    
    # Protocol Selection
    st.markdown("---")
    st.markdown("#### 🔐 Protocol Selection")
    protocol = st.radio(
        "Select protocol for new authentications:",
        ["HYBRID_PQC", "FULL_PQC", "CLASSICAL"],
        index=0,
        key="protocol_selector"
    )
    
    if st.button("🔄 Switch Protocol", use_container_width=True, key="switch_protocol"):
        success = st.session_state.base_station.switch_protocol(protocol)
        if success:
            add_log(f"PROTOCOL SWITCHED to {protocol}")
            st.rerun()
    
    # Simulation Controls
    st.markdown("---")
    st.markdown("#### ⏯️ Simulation Controls")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("⏸️ Pause" if st.session_state.simulation_running else "▶️ Resume",
                    use_container_width=True, key="pause_resume"):
            st.session_state.simulation_running = not st.session_state.simulation_running
            add_log(f"Simulation {'paused' if not st.session_state.simulation_running else 'resumed'}")
            st.rerun()
    
    with col_c2:
        if st.button("🔄 Reset All", use_container_width=True, key="reset_all"):
            st.session_state.drones = create_initial_drones()
            st.session_state.base_station = BaseStation()
            st.session_state.logs = []
            st.session_state.communications = []
            st.session_state.simulation_running = True
            st.session_state.simulation_start_time = time.time()
            add_log("Simulation completely reset")
            st.rerun()
    
    # Evaluation Tools
    st.markdown("---")
    st.markdown("#### 📊 Evaluation Tools")
    
    if st.button("📈 Run Time-based Comparison", use_container_width=True, key="run_comparison"):
        # Create test drones and run all protocols
        test_drones = []
        for proto in ["CLASSICAL", "FULL_PQC", "HYBRID_PQC"]:
            test_drone = Drone(f"TEST_{proto}", f"Test_{proto}", "test", "dummy.npz", model)
            success, metrics, _ = st.session_state.base_station.authenticate_drone(
                test_drone, force_protocol=proto
            )
            if success:
                test_drones.append(f"{proto}: {metrics.handshake_time_ms:.1f}ms")
        
        add_log(f"TIME-BASED COMPARISON: {' | '.join(test_drones)}", "EVAL")
    
    # Drone List
    st.markdown("---")
    st.markdown("#### 🛸 Active Drones List")
    
    for drone in st.session_state.drones:
        # Determine card style
        if drone.pqc_authenticated:
            if drone.protocol_used == "CLASSICAL":
                card_class = "protocol-card classical-card"
                emoji = "🔑"
            elif drone.protocol_used == "FULL_PQC":
                card_class = "protocol-card full-pqc-card"
                emoji = "⚛️"
            else:
                card_class = "protocol-card hybrid-pqc-card"
                emoji = "🔐"
        else:
            card_class = "protocol-card"
            emoji = "⏳"
        
        with st.container():
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col_d1, col_d2, col_d3 = st.columns([2, 2, 1])
            with col_d1:
                st.markdown(f"**{emoji} {drone.label}**")
                st.caption(drone.model_name)
            
            with col_d2:
                if drone.last_verification_distance:
                    st.caption(f"Score: {drone.last_verification_distance:.3f}")
                elif drone.pqc_authenticated and drone.auth_metrics:
                    st.caption(f"Auth: {drone.auth_metrics.handshake_time_ms:.1f}ms")
            
            with col_d3:
                if st.button("👁️", key=f"view_{drone.id}", help=f"View {drone.label}"):
                    st.session_state.selected_drone = drone.id
            
            st.markdown('</div>', unsafe_allow_html=True)

# =======================
# RIGHT PANEL - VISUALIZATION & ANALYSIS
# =======================
with col2:
    # Update simulation
    update_simulation()
    
    # Top section: Visualization and Time-based metrics
    col_viz, col_time = st.columns([2, 1])
    
    with col_viz:
        st.markdown("### 🌐 Swarm Visualization")
        swarm_fig = create_swarm_visualization()
        st.pyplot(swarm_fig, use_container_width=True)
        plt.close(swarm_fig)
    
    with col_time:
        st.markdown("### ⏱️ Time Statistics")
        simulation_time = time.time() - st.session_state.simulation_start_time
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Elapsed Time", f"{simulation_time:.1f}s")
            authenticated = sum(1 for d in st.session_state.drones if d.pqc_authenticated)
            st.metric("Authenticated", f"{authenticated}/{len(st.session_state.drones)}")
        
        with col_t2:
            operations = (len(st.session_state.base_station.evaluator.classical_metrics) +
                         len(st.session_state.base_station.evaluator.full_pqc_metrics) +
                         len(st.session_state.base_station.evaluator.hybrid_pqc_metrics))
            st.metric("Total Ops", operations)
            if simulation_time > 0:
                st.metric("Ops/sec", f"{operations/simulation_time:.2f}")
    
    # Middle section: Time-based evaluation charts
    st.markdown("---")
    st.markdown("### 📊 Time-based Protocol Evaluation")
    
    eval_tabs = st.tabs(["⏱️ Time Evolution", "📈 Efficiency Analysis", "📋 Detailed Metrics"])
    
    with eval_tabs[0]:
        # Time Evolution Charts
        if (st.session_state.base_station.evaluator.classical_metrics or 
            st.session_state.base_station.evaluator.full_pqc_metrics or 
            st.session_state.base_station.evaluator.hybrid_pqc_metrics):
            
            time_fig = st.session_state.base_station.evaluator.plot_time_based_comparison()
            st.pyplot(time_fig, use_container_width=True)
            plt.close(time_fig)
        else:
            st.info("Run authentication tests to see time-based evolution")
    
    with eval_tabs[1]:
        # Efficiency Analysis
        st.markdown("#### ⚡ Hybrid PQC Efficiency Over Time")
        
        efficiency_df = st.session_state.base_station.evaluator.get_hybrid_efficiency_over_time()
        if efficiency_df is not None and not efficiency_df.empty:
            # Plot efficiency improvements
            fig_eff, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Handshake Improvement
            axes[0, 0].plot(efficiency_df['Time'], efficiency_df['Handshake Improvement %'], 
                          'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Improvement %')
            axes[0, 0].set_title('Handshake Time Improvement (Hybrid vs Full PQC)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Data Encryption Improvement
            axes[0, 1].plot(efficiency_df['Time'], efficiency_df['Data Encryption Improvement %'], 
                          'r-s', linewidth=2, markersize=4)
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Improvement %')
            axes[0, 1].set_title('Data Encryption Improvement (Hybrid vs Full PQC)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Total Improvement
            axes[1, 0].plot(efficiency_df['Time'], efficiency_df['Total Improvement %'], 
                          'g-^', linewidth=2, markersize=4)
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Improvement %')
            axes[1, 0].set_title('Total Time Improvement (Hybrid vs Full PQC)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Absolute Time Comparison
            axes[1, 1].plot(efficiency_df['Time'], efficiency_df['Full PQC Total (ms)'], 
                          'r--', label='Full PQC', linewidth=2)
            axes[1, 1].plot(efficiency_df['Time'], efficiency_df['Hybrid PQC Total (ms)'], 
                          'b-', label='Hybrid PQC', linewidth=2)
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Total Time (ms)')
            axes[1, 1].set_title('Absolute Time Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_eff)
            plt.close(fig_eff)
            
            # Show efficiency statistics
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                avg_handshake_imp = efficiency_df['Handshake Improvement %'].mean()
                st.metric("Avg Handshake Imp", f"{avg_handshake_imp:.1f}%")
            with col_e2:
                avg_data_imp = efficiency_df['Data Encryption Improvement %'].mean()
                st.metric("Avg Data Imp", f"{avg_data_imp:.1f}%")
            with col_e3:
                avg_total_imp = efficiency_df['Total Improvement %'].mean()
                st.metric("Avg Total Imp", f"{avg_total_imp:.1f}%")
    
    with eval_tabs[2]:
        # Detailed Time-based Metrics
        time_stats = st.session_state.base_station.evaluator.get_time_based_stats()
        if time_stats is not None and not time_stats.empty:
            st.dataframe(time_stats, use_container_width=True, hide_index=True)
        else:
            st.info("No detailed time metrics available yet")

# =======================
# BOTTOM PANEL - RF DATA VISUALIZATION
# =======================
if st.session_state.selected_drone:
    selected_drone = next((d for d in st.session_state.drones 
                          if d.id == st.session_state.selected_drone), None)
    
    if selected_drone:
        st.markdown("---")
        st.markdown(f"### 📡 RF Data Analysis: {selected_drone.label}")
        
        # Create tabs for different RF analyses
        rf_tabs = st.tabs(["📊 IQ Data Analysis", "🎵 Spectrogram Analysis", "🔍 RF Fingerprint", "📈 Protocol Performance"])
        
        with rf_tabs[0]:
            # IQ Data Analysis
            st.markdown(f"#### 📶 IQ Signal Analysis: {selected_drone.label}")
            
            if selected_drone.iq_data is not None:
                iq_fig = plot_comprehensive_iq_analysis(selected_drone.iq_data, selected_drone.label)
                st.pyplot(iq_fig, use_container_width=True)
                plt.close(iq_fig)
                
                # IQ Statistics
                col_iq1, col_iq2, col_iq3, col_iq4 = st.columns(4)
                with col_iq1:
                    st.metric("Samples", len(selected_drone.iq_data))
                with col_iq2:
                    avg_power = np.mean(np.abs(selected_drone.iq_data)**2)
                    st.metric("Avg Power", f"{avg_power:.4f}")
                with col_iq3:
                    i_mean = np.mean(np.real(selected_drone.iq_data))
                    st.metric("I Mean", f"{i_mean:.4f}")
                with col_iq4:
                    q_mean = np.mean(np.imag(selected_drone.iq_data))
                    st.metric("Q Mean", f"{q_mean:.4f}")
            else:
                st.warning("No IQ data available for this drone")
        
        with rf_tabs[1]:
            # Spectrogram Analysis
            st.markdown(f"#### 🎵 Spectrogram Analysis: {selected_drone.label}")
            
            if selected_drone.spectrogram_data is not None:
                spec_fig = plot_enhanced_spectrogram(selected_drone.spectrogram_data, selected_drone.label)
                st.pyplot(spec_fig, use_container_width=True)
                plt.close(spec_fig)
                
                # Spectrogram Statistics
                col_spec1, col_spec2, col_spec3 = st.columns(3)
                with col_spec1:
                    st.metric("Frequency Bins", selected_drone.spectrogram_data.shape[0])
                with col_spec2:
                    st.metric("Time Samples", selected_drone.spectrogram_data.shape[1])
                with col_spec3:
                    max_freq = np.max(selected_drone.spectrogram_data)
                    st.metric("Max Power", f"{max_freq:.2f}")
            else:
                st.warning("No spectrogram data available for this drone")
        
        with rf_tabs[2]:
            # RF Fingerprint Comparison
            st.markdown(f"#### 🔍 RF Fingerprint: {selected_drone.label}")
            
            # Let user select another drone for comparison
            other_drones = [d for d in st.session_state.drones if d.id != selected_drone.id]
            if other_drones:
                comparison_options = ["None"] + [d.label for d in other_drones]
                selected_comparison = st.selectbox(
                    "Compare with another drone:",
                    comparison_options,
                    key="comparison_select"
                )
                
                if selected_comparison != "None":
                    comparison_drone = next((d for d in other_drones if d.label == selected_comparison), None)
                    if comparison_drone:
                        if (selected_drone.iq_data is not None and 
                            comparison_drone.iq_data is not None):
                            comp_fig = plot_rf_fingerprint_comparison(selected_drone, comparison_drone)
                            if comp_fig:
                                st.pyplot(comp_fig, use_container_width=True)
                                plt.close(comp_fig)
                                
                                # Calculate similarity metrics
                                if (len(selected_drone.iq_data) > 0 and 
                                    len(comparison_drone.iq_data) > 0):
                                    min_len = min(len(selected_drone.iq_data), 
                                                 len(comparison_drone.iq_data))
                                    iq1 = selected_drone.iq_data[:min_len]
                                    iq2 = comparison_drone.iq_data[:min_len]
                                    
                                    # Calculate correlations
                                    i_corr = np.corrcoef(np.real(iq1), np.real(iq2))[0, 1]
                                    q_corr = np.corrcoef(np.imag(iq1), np.imag(iq2))[0, 1]
                                    mag_corr = np.corrcoef(np.abs(iq1), np.abs(iq2))[0, 1]
                                    
                                    col_comp1, col_comp2, col_comp3 = st.columns(3)
                                    with col_comp1:
                                        st.metric("I Correlation", f"{i_corr:.3f}")
                                    with col_comp2:
                                        st.metric("Q Correlation", f"{q_corr:.3f}")
                                    with col_comp3:
                                        st.metric("Mag Correlation", f"{mag_corr:.3f}")
                        else:
                            st.warning("One or both drones missing IQ data for comparison")
                else:
                    st.info("Select another drone to compare RF fingerprints")
            else:
                st.info("No other drones available for comparison")
        
        with rf_tabs[3]:
            # Protocol Performance for this drone
            st.markdown(f"#### 📈 Authentication Performance: {selected_drone.label}")
            
            if selected_drone.pqc_authenticated and selected_drone.auth_metrics:
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                with col_perf1:
                    st.metric("Protocol Used", selected_drone.protocol_used)
                with col_perf2:
                    st.metric("Handshake Time", f"{selected_drone.auth_metrics.handshake_time_ms:.1f}ms")
                with col_perf3:
                    st.metric("Data Encryption", f"{selected_drone.auth_metrics.encryption_time_ms:.1f}ms")
                
                # Show when authentication happened
                if selected_drone.auth_timestamp:
                    st.write(f"**Authentication Time:** {selected_drone.auth_timestamp:.1f} seconds after simulation start")
                
                # Show protocol-specific details
                if selected_drone.protocol_used == "HYBRID_PQC":
                    st.success("✅ **Hybrid PQC Advantages:**")
                    st.write("- Quantum-resistant handshake (PQC)")
                  
                elif selected_drone.protocol_used == "FULL_PQC":
                    st.warning("⚠️ **Full PQC Characteristics:**")
                    st.write("- Fully quantum-resistant")
                 
                else:  # CLASSICAL
                    st.error("❌ **Classical Protocol Limitations:**")
                    st.write("- Vulnerable to quantum attacks")
                  
            else:
                st.info("This drone has not been authenticated yet")

# =======================
# LOGS PANEL (Bottom of page)
# =======================
st.markdown("---")
st.markdown("### 📜 System Logs")

log_container = st.container(height=250)
with log_container:
    # Display logs with time-based context
    simulation_time = time.time() - st.session_state.simulation_start_time
    st.caption(f"Simulation running for {simulation_time:.1f} seconds")
    
    for log in reversed(st.session_state.logs[-20:]):
        if "TIME" in log or "⏱️" in log:
            st.markdown(f'<div style="background: rgba(255, 215, 0, 0.1); padding: 8px; margin: 4px 0; border-left: 4px solid #FFD700; border-radius: 4px;">{log}</div>', 
                       unsafe_allow_html=True)
        elif "RF" in log or "📶" in log:
            st.markdown(f'<div style="background: rgba(0, 206, 209, 0.1); padding: 8px; margin: 4px 0; border-left: 4px solid #00CED1; border-radius: 4px;">{log}</div>', 
                       unsafe_allow_html=True)
        elif "AUTH" in log or "🔐" in log or "🔑" in log or "⚛️" in log:
            st.markdown(f'<div style="background: rgba(147, 112, 219, 0.1); padding: 8px; margin: 4px 0; border-left: 4px solid #9370DB; border-radius: 4px;">{log}</div>', 
                       unsafe_allow_html=True)
        elif "ADDED" in log or "🆕" in log:
            st.markdown(f'<div style="background: rgba(50, 205, 50, 0.1); padding: 8px; margin: 4px 0; border-left: 4px solid #32CD32; border-radius: 4px;">{log}</div>', 
                       unsafe_allow_html=True)
        elif "MALICIOUS" in log or "🚨" in log:
            st.markdown(f'<div style="background: rgba(255, 0, 0, 0.1); padding: 8px; margin: 4px 0; border-left: 4px solid #FF0000; border-radius: 4px;">{log}</div>', 
                       unsafe_allow_html=True)
        else:
            st.write(log)

# Auto-refresh
if st.session_state.simulation_running:
    time.sleep(0.05)
    st.rerun()