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
from scipy import signal
import pandas as pd


# =======================
# PHASE-2 : HYBRID PQC (SIMULATED)
# =======================
import secrets
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


class HybridPQC:
    @staticmethod
    def quantum_handshake(drone_id):
        """
        Simulated PQC handshake (Kyber-style)
        """
        start = time.time()

        # Simulated lattice-based secret
        shared_secret = secrets.token_bytes(32)

        # Derive AES key
        aes_key = hashlib.sha256(shared_secret).digest()

        latency = time.time() - start
        bandwidth = len(shared_secret)

        return aes_key, latency, bandwidth


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
# CONFIG
# -----------------------
DATASET_ROOT = r"F:\K DRIVE\MIT_Learnings\Sem8\dataset\code\sei_dataset4"
LOG_FILE_PATH = "uav_simulation_logs.txt"


# Verify dataset exists
if not os.path.exists(DATASET_ROOT):
    st.error(f"Dataset path not found: {DATASET_ROOT}")
    st.stop()

MAX_DRONES = 6
INITIAL_DRONES = 3
WORLD_SIZE = 1.0  # 1x1 world
COMM_RANGE = 0.50  # Communication range (25% of world size)
THRESHOLD = 0.6  # Trust threshold for verification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colors
COLORS = {
    'trusted': '#00FF00',  # Green
    'malicious': '#FF0000',  # Red
    'communication': '#4169E1',  # Royal Blue
    'isolated': '#FFA500',  # Orange
    'background': '#0E1117',  # Streamlit dark background
    'text': '#FFFFFF',
    'enemy': '#FF6B6B',
    'unknown': '#FFD700'
}

# -----------------------
# LOAD MODEL (ONCE)
# -----------------------
@st.cache_resource
def load_model():
    try:
        model_paths = [
            "../../PTH/lsnet_epoch_20_final.pth",
            "lsnet_epoch_20_final.pth",
            "./PTH/lsnet_epoch_20_final.pth",
            "checkpoints/lsnet_epoch_20_final.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                st.success(f"Found model at: {model_path}")
                break
        
        if model_path is None:
            # Try to find any .pth file
            import glob
            pth_files = glob.glob("**/*.pth", recursive=True)
            if pth_files:
                model_path = pth_files[0]
                st.success(f"Using model: {model_path}")
            else:
                st.warning("No model file found. Using random weights for demonstration.")
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
        st.error(f"Error loading model: {e}")
        st.warning("Using random weights for demonstration")
        model = LSNet()
        model.to(DEVICE)
        model.eval()
        return model

model = load_model()

# -----------------------
# DRONE CLASS
# -----------------------
class Drone:
    def __init__(self, drone_id, label, model_name, npz_path, model_instance):
        self.id = drone_id
        self.label = label
        self.model_name = model_name  # e.g., 'dji', 'futaba', 'enemy', 'unknown'
        self.npz_path = npz_path
        self.model = model_instance
        
        # Position in world - start from edges to encourage movement toward center
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            self.pos = np.array([random.uniform(0.1, WORLD_SIZE-0.1), WORLD_SIZE-0.1])
        elif side == 'bottom':
            self.pos = np.array([random.uniform(0.1, WORLD_SIZE-0.1), 0.1])
        elif side == 'left':
            self.pos = np.array([0.1, random.uniform(0.1, WORLD_SIZE-0.1)])
        else:  # right
            self.pos = np.array([WORLD_SIZE-0.1, random.uniform(0.1, WORLD_SIZE-0.1)])
        
        # Initial direction - slightly toward center
        center = np.array([WORLD_SIZE/2, WORLD_SIZE/2])
        self.direction = center - self.pos
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        
        # Add some randomness to direction
        self.direction += np.random.randn(2) * 0.2
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        
        # Drone properties
        self.trusted = True if 'enemy' not in model_name.lower() else False
        self.isolated = False
        self.communication_history = []
        self.speed = 0.008 if 'enemy' in model_name.lower() else 0.005
        
        # Data for visualization
        self.iq_data = None
        self.spectrogram_data = None
        self.last_verification_result = None
        self.last_verification_distance = None
        
        # Load sample data
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load IQ data and spectrogram from NPZ file"""
        try:
            data = np.load(self.npz_path, allow_pickle=True)
            
            # Try different possible keys
            iq_keys = ['iq_data', 'iq', 'signal', 'samples']
            spec_keys = ['spectrogram', 'spec', 'spectro', 'stft']
            
            # Find IQ data
            for key in iq_keys:
                if key in data:
                    self.iq_data = data[key]
                    break
            
            # Find spectrogram data
            for key in spec_keys:
                if key in data:
                    self.spectrogram_data = data[key]
                    break
            
            # If no spectrogram but have IQ data, create one
            if self.spectrogram_data is None and self.iq_data is not None:
                # Take first 1024 samples for spectrogram
                iq_samples = self.iq_data[:1024] if len(self.iq_data) > 1024 else self.iq_data
                if isinstance(iq_samples, np.ndarray) and iq_samples.dtype == np.complex128:
                    f, t, Sxx = signal.spectrogram(iq_samples, fs=1000, nperseg=64, noverlap=32)
                    self.spectrogram_data = Sxx
            
            # Create dummy data if still None
            if self.iq_data is None:
                self.iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
            
            if self.spectrogram_data is None:
                t = np.linspace(0, 1, 100)
                f = np.linspace(0, 500, 64)
                self.spectrogram_data = np.random.rand(64, 100)
                
        except Exception as e:
            st.warning(f"Could not load data for {self.label}: {e}")
            # Create dummy data for visualization
            self.iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
            t = np.linspace(0, 1, 100)
            f = np.linspace(0, 500, 64)
            self.spectrogram_data = np.random.rand(64, 100)
    
    def move(self):
        """Move drone in swarm-like pattern"""
        if self.isolated:
            # Isolated drones move randomly and slowly
            self.direction += np.random.randn(2) * 0.15
            speed = self.speed * 0.5
        else:
            # Normal swarm movement with some randomness
            self.direction += np.random.randn(2) * 0.08
            speed = self.speed
        
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        
        # Update position
        self.pos += self.direction * speed
        
        # Bounce off walls with slight random angle
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
        
        # Keep direction normalized
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
    
    def verify_other_drone(self, other_drone):
        """Verify another drone using LSNet model"""
        try:
            # Load samples
            data_a = np.load(self.npz_path, allow_pickle=True)
            data_b = np.load(other_drone.npz_path, allow_pickle=True)
            
            # Find spectrogram keys
            spec_keys = ['spectrogram', 'spec', 'spectro', 'stft']
            spec_a = None
            spec_b = None
            
            for key in spec_keys:
                if key in data_a:
                    spec_a = data_a[key]
                    break
            
            for key in spec_keys:
                if key in data_b:
                    spec_b = data_b[key]
                    break
            
            # If no spectrogram found, use dummy data
            if spec_a is None:
                spec_a = np.random.randn(64, 64)
            if spec_b is None:
                spec_b = np.random.randn(64, 64)
            
            # Ensure spectrograms are 2D
            if spec_a.ndim > 2:
                spec_a = spec_a.squeeze()
            if spec_b.ndim > 2:
                spec_b = spec_b.squeeze()
            
            # Stack two copies to create 2-channel input
            x = torch.tensor(np.stack([spec_a, spec_a]), dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(np.stack([spec_b, spec_b]), dtype=torch.float32).unsqueeze(0)
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Get embeddings
            with torch.no_grad():
                za = model(x)
                zb = model(y)
            
            # Calculate cosine distance (since embeddings are normalized)
            distance = 1.0 - torch.nn.functional.cosine_similarity(za, zb).item()
            
            # Determine if trusted based on threshold
            is_trusted = distance < THRESHOLD
            
            # Store result
            self.last_verification_result = is_trusted
            self.last_verification_distance = distance
            
            return is_trusted, distance
            
        except Exception as e:
            st.error(f"Verification error between {self.label} and {other_drone.label}: {e}")
            # Return random result for demonstration
            distance = random.uniform(0.5, 2.0)
            is_trusted = distance < THRESHOLD
            return is_trusted, distance
    
    def start_pqc_session(self, base_station):
        latency, bandwidth = base_station.accept_drone(self)

        self.communication_history.append(
            f"PQC Session Established | Latency: {latency:.4f}s | BW: {bandwidth}B"
        )

        return latency * 1000, bandwidth / 1024



# =======================
# BASE STATION
# =======================
class BaseStation:
    def __init__(self):
        self.sessions = {}
        self.logs = []
        self.metrics = {
            "classical": [],
            "hybrid": [],
            "quantum": []
        }

    def accept_drone(self, drone):
        aes_key, latency, bandwidth = HybridPQC.quantum_handshake(drone.id)

        self.sessions[drone.id] = aes_key

        log = f"PQC HANDSHAKE SUCCESS | {drone.label} | {latency:.4f}s | {bandwidth}B"
        self.logs.append(log)

         # Store metrics dynamically
                # ---- realistic system-level modeling ----
        NETWORK_DELAY_MS = 3.5          # satellite / UAV uplink
        PQC_COMPUTE_MS = latency * 1000 # real compute
        PROTOCOL_OVERHEAD_MS = 4.0      # handshake frames

        latency_ms = NETWORK_DELAY_MS + PQC_COMPUTE_MS + PROTOCOL_OVERHEAD_MS

        # PQC keys + headers + certs
        bandwidth_kb = (bandwidth + 1024) / 1024   # add 1 KB protocol overhead

        self.metrics["hybrid"].append({
            "drone": drone.id,
            "latency": latency_ms,
            "bandwidth": bandwidth_kb,
            "security": 0.95,
            "swarm_size": len(st.session_state.drones),
            "time": time.time()
         })


        return latency_ms, bandwidth_kb

    def receive_update(self, drone, payload_size=2048):
        aes_key = self.sessions[drone.id]

        start = time.time()
        cipher = AES.new(aes_key, AES.MODE_GCM)
        encrypted = cipher.encrypt(get_random_bytes(payload_size))
        latency = time.time() - start

        return latency, len(encrypted)

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def pick_random_drone():
    """Pick a random drone model and sample from dataset"""
    try:
        # Get all model folders
        models = [d for d in os.listdir(DATASET_ROOT) 
                 if os.path.isdir(os.path.join(DATASET_ROOT, d))]
        
        if not models:
            st.error("No drone models found in dataset")
            # Return dummy data
            return "simulated", "dummy_path.npz"
        
        # Pick random model (weighted to include enemies/unknown)
        model_weights = []
        for m in models:
            if 'enemy' in m.lower():
                model_weights.append(0.3)  # 30% chance for enemy
            elif 'unknown' in m.lower():
                model_weights.append(0.2)  # 20% chance for unknown
            else:
                model_weights.append(0.1)  # 10% chance for others
        
        # Normalize weights
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w/total_weight for w in model_weights]
            model_name = random.choices(models, weights=model_weights)[0]
        else:
            model_name = random.choice(models)
        
        model_path = os.path.join(DATASET_ROOT, model_name)
        
        # Get all NPZ files in model folder
        npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
        
        if not npz_files:
            # Create dummy path
            npz_path = f"dummy_{model_name}.npz"
        else:
            # Pick random NPZ file
            npz_file = random.choice(npz_files)
            npz_path = os.path.join(model_path, npz_file)
        
        return model_name, npz_path
        
    except Exception as e:
        st.error(f"Error picking drone: {e}")
        return "simulated", "dummy_path.npz"

def create_initial_drones():
    """Create initial drones for simulation"""
    drones = []
    models_to_use = ['dji', 'futaba', 'enemy']  # Start with these models
    
    for i in range(INITIAL_DRONES):
        # For initial drones, use specific models
        if i < len(models_to_use):
            model_name = models_to_use[i]
            # Find the model folder
            model_path = os.path.join(DATASET_ROOT, model_name)
            if os.path.exists(model_path):
                npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
                if npz_files:
                    npz_file = random.choice(npz_files)
                    npz_path = os.path.join(model_path, npz_file)
                else:
                    # Fallback to random if no files
                    model_name, npz_path = pick_random_drone()
            else:
                # Fallback to random if model doesn't exist
                model_name, npz_path = pick_random_drone()
        else:
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


# =======================
# BASELINE METRIC SIMULATION (C)
# =======================
def simulate_baselines(base_station, swarm_size):
    base_station.metrics["classical"] = [{
        "latency": 2 + 0.3 * swarm_size,
        "bandwidth": 1 + 0.2 * swarm_size,
        "security": 0.6
    }]

    base_station.metrics["quantum"] = [{
        "latency": 8 + 0.5 * swarm_size,
        "bandwidth": 4 + 0.6 * swarm_size,
        "security": 1.0
    }]




def plot_iq_data(iq_data, title):
    """Plot IQ data"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # Ensure we have data
    if iq_data is None:
        axes[0].text(0.5, 0.5, 'No IQ Data', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No IQ Data', ha='center', va='center')
        axes[2].text(0.5, 0.5, 'No IQ Data', ha='center', va='center')
    else:
        # Take first 500 samples
        samples_to_plot = min(500, len(iq_data))
        
        # Plot I component
        axes[0].plot(np.real(iq_data)[:samples_to_plot], color='cyan', linewidth=0.8)
        axes[0].set_title(f'{title} - I Component', fontsize=10)
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot Q component
        axes[1].plot(np.imag(iq_data)[:samples_to_plot], color='magenta', linewidth=0.8)
        axes[1].set_title(f'{title} - Q Component', fontsize=10)
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Plot constellation (first 200 samples)
        const_samples = min(200, len(iq_data))
        axes[2].scatter(np.real(iq_data)[:const_samples], 
                       np.imag(iq_data)[:const_samples], 
                       alpha=0.6, s=10, c='lime', edgecolors='black', linewidths=0.2)
        axes[2].set_title(f'{title} - Constellation', fontsize=10)
        axes[2].set_xlabel('I')
        axes[2].set_ylabel('Q')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_spectrogram(spectrogram_data, title):
    """Plot spectrogram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if spectrogram_data is not None and spectrogram_data.ndim == 2:
        # Plot combined spectrogram
        im1 = ax1.imshow(10 * np.log10(spectrogram_data + 1e-10), 
                        aspect='auto', origin='lower', cmap='hot',
                        interpolation='bilinear')
        ax1.set_title(f'{title} - Combined Spectrogram', fontsize=10)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # Plot magnitude spectrogram
        magnitude = np.abs(spectrogram_data)
        im2 = ax2.imshow(magnitude, aspect='auto', origin='lower', cmap='viridis',
                        interpolation='bilinear')
        ax2.set_title(f'{title} - Magnitude Spectrogram', fontsize=10)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency')
        plt.colorbar(im2, ax=ax2, label='Magnitude')
    else:
        ax1.text(0.5, 0.5, 'Invalid spectrogram data', 
                ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'Invalid spectrogram data', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig

# -----------------------
# SESSION STATE INITIALIZATION
# -----------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Simulation state
        st.session_state.simulation_running = True
        st.session_state.pause_time = 0
        st.session_state.last_update = time.time()
        
        # Drone management
        st.session_state.drones = create_initial_drones()
        st.session_state.selected_drone = None

        # Base station
        st.session_state.base_station = BaseStation()
        simulate_baselines(st.session_state.base_station, len(st.session_state.drones))
        # Logging
        st.session_state.logs = []
        st.session_state.max_logs = 100
        
        # Communication history
        st.session_state.communications = []
        st.session_state.malicious_drones = set()
        
        # Add initial log
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] INFO: Simulation started with {INITIAL_DRONES} drones")
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] INFO: Trust threshold set to {THRESHOLD}")
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] INFO: Communication range: {COMM_RANGE}")

initialize_session_state()

# -----------------------
# LOGGING FUNCTIONS
# -----------------------
def add_log(message, level="INFO"):
    """Add a log message"""
    timestamp = time.strftime("%H:%M:%S")
    
    # Add emoji based on level
    if level == "WARNING":
        emoji = "⚠️"
    elif level == "ERROR":
        emoji = "❌"
    elif "MALICIOUS" in message:
        emoji = "🚨"
    elif "TRUSTED" in message:
        emoji = "✅"
    elif "COMM" in message or "communicat" in message.lower():
        emoji = "📡"
    elif "ADDED" in message:
        emoji = "🆕"
    else:
        emoji = "ℹ️"
    
    log_entry = f"[{timestamp}] {emoji} {message}"
    st.session_state.logs.append(log_entry)
    write_log_to_file(log_entry)
    
    # Keep logs within limit
    if len(st.session_state.logs) > st.session_state.max_logs:
        st.session_state.logs = st.session_state.logs[-st.session_state.max_logs:]

# Append log to file

def write_log_to_file(log_entry):
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")



# -----------------------
# SIMULATION FUNCTIONS
# -----------------------
def update_simulation():
    """Update drone positions and handle communications"""
    if not st.session_state.simulation_running:
        return
    
    current_time = time.time()
    
    # Only update every 0.1 seconds to control speed
    if current_time - st.session_state.last_update < 0.1:
        return
    
    st.session_state.last_update = current_time
    
    # Move all drones
    for drone in st.session_state.drones:
        drone.move()
    
    # Clear previous communications
    st.session_state.communications = []
    
    # Check for communications between all drone pairs
    for i in range(len(st.session_state.drones)):
        for j in range(i + 1, len(st.session_state.drones)):
            drone_a = st.session_state.drones[i]
            drone_b = st.session_state.drones[j]
            
            # Skip if either drone is isolated
            if drone_a.isolated or drone_b.isolated:
                continue
            
            # Calculate distance
            distance = euclidean(drone_a.pos, drone_b.pos)
            
            # If drones are in communication range
            if distance < COMM_RANGE:
                # Record communication
                comm_id = f"{drone_a.id}-{drone_b.id}"
                st.session_state.communications.append((drone_a, drone_b, distance, comm_id))
                
                # Only verify if both drones are trusted (not yet marked malicious)
                if drone_a.trusted and drone_b.trusted:
                    # Verify each other (bidirectional verification)
                    trusted_a_to_b, dist_a_to_b = drone_a.verify_other_drone(drone_b)
                    trusted_b_to_a, dist_b_to_a = drone_b.verify_other_drone(drone_a)
                    
                    # Use the maximum distance for decision (most conservative)
                    max_distance = max(dist_a_to_b, dist_b_to_a)
                    is_trusted = max_distance < THRESHOLD
                    
                    # Log the communication
                    log_msg = f"COMM: {drone_a.label} ↔ {drone_b.label} | "
                    log_msg += f"Physical: {distance:.3f} | "
                    log_msg += f"Trust Score: {max_distance:.3f}"
                    add_log(log_msg)


                    if is_trusted:
                        # Drone A → Base Station
                        if not hasattr(drone_a, "pqc_done") or not drone_a.pqc_done:
                            latency, bw = st.session_state.base_station.accept_drone(drone_a)
                            drone_a.pqc_done = True

                            add_log(
                                f"PQC SESSION: {drone_a.label} → BaseStation | "
                                f"Latency: {latency:.4f}s | BW: {bw}B"
                            )

                        # Drone B → Base Station
                        if not hasattr(drone_b, "pqc_done") or not drone_b.pqc_done:
                            latency, bw = st.session_state.base_station.accept_drone(drone_b)
                            drone_b.pqc_done = True

                            add_log(
                                f"PQC SESSION: {drone_b.label} → BaseStation | "
                                f"Latency: {latency:.4f}s | BW: {bw}B"
                            )

                    
                    # If verification fails, mark as malicious
                    if not is_trusted:
                        # Mark the drone with higher distance as malicious
                        if dist_a_to_b >= dist_b_to_a:
                            malicious_drone = drone_b
                            detector_drone = drone_a
                        else:
                            malicious_drone = drone_a
                            detector_drone = drone_b
                        
                        malicious_drone.trusted = False
                        malicious_drone.isolated = True
                        st.session_state.malicious_drones.add(malicious_drone.id)
                        
                        # Add to communication history
                        malicious_drone.communication_history.append(
                            f"Marked malicious by {detector_drone.label} (dist: {max_distance:.3f})"
                        )
                        
                        add_log(f"MALICIOUS DETECTED: {malicious_drone.label} by {detector_drone.label}", "WARNING")
                        
                        # Remove communications involving this drone
                        st.session_state.communications = [
                            comm for comm in st.session_state.communications
                            if malicious_drone not in [comm[0], comm[1]]
                        ]
                
                # If one drone is already malicious, isolate communication
                elif not drone_a.trusted or not drone_b.trusted:
                    # Still show communication line but with warning
                    add_log(f"BLOCKED: {drone_a.label} ↔ {drone_b.label} | One drone is malicious", "WARNING")

# -----------------------
# VISUALIZATION FUNCTIONS
# -----------------------
def create_swarm_visualization():
    """Create the swarm visualization plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set plot limits with some margin
    margin = 0.05
    ax.set_xlim(-margin, WORLD_SIZE + margin)
    ax.set_ylim(-margin, WORLD_SIZE + margin)
    
    # Set background color
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    
    # Draw communication range circles for trusted drones
    for drone in st.session_state.drones:
        if not drone.isolated:
            circle = plt.Circle(drone.pos, COMM_RANGE, 
                              color='blue', alpha=0.03, fill=True, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
    
    # Draw communication lines with arrows
    for comm in st.session_state.communications:
        drone_a, drone_b, distance, comm_id = comm
        
        # Determine line style based on trust status
        if drone_a.trusted and drone_b.trusted:
            line_color = 'cyan'
            line_alpha = 0.7
            line_style = '-'
            line_width = 1.5
        else:
            line_color = 'red'
            line_alpha = 0.4
            line_style = '--'
            line_width = 1.0
        
        # Draw arrow line
        ax.annotate("", xy=drone_b.pos, xytext=drone_a.pos,
                   arrowprops=dict(arrowstyle="->", 
                                  color=line_color,
                                  lw=line_width,
                                  alpha=line_alpha,
                                  linestyle=line_style))
        
        # Add distance label at midpoint
        mid_point = (drone_a.pos + drone_b.pos) / 2
        offset = np.array([0.01, 0.01])  # Small offset for label
        ax.text(mid_point[0] + offset[0], mid_point[1] + offset[1], 
               f"{distance:.2f}", fontsize=7, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor='none'),
               color='white')
    
    # Draw drones
    for drone in st.session_state.drones:
        # Determine color and size based on status
        if drone.isolated:
            face_color = COLORS['isolated']
            edge_color = 'red'
            line_width = 3
            size = 350
            marker = 'X'
            alpha = 0.8
        elif not drone.trusted:
            face_color = COLORS['malicious']
            edge_color = 'darkred'
            line_width = 2
            size = 300
            marker = 'o'
            alpha = 1.0
        elif 'enemy' in drone.model_name.lower():
            face_color = COLORS['enemy']
            edge_color = 'darkorange'
            line_width = 2
            size = 280
            marker = 'o'
            alpha = 0.9
        elif 'unknown' in drone.model_name.lower():
            face_color = COLORS['unknown']
            edge_color = 'gold'
            line_width = 2
            size = 280
            marker = 'o'
            alpha = 0.9
        else:
            face_color = COLORS['trusted']
            edge_color = 'darkgreen'
            line_width = 2
            size = 250
            marker = 'o'
            alpha = 1.0
        
        # Draw drone
        ax.scatter(drone.pos[0], drone.pos[1], 
                  s=size, c=face_color, marker=marker,
                  edgecolors=edge_color, linewidths=line_width, alpha=alpha, zorder=10)
        
        # Add drone label
        label_bg_color = 'black' if not drone.isolated else 'darkred'
        ax.text(drone.pos[0], drone.pos[1] - 0.035, 
               drone.label, fontsize=9, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=label_bg_color, 
                        alpha=0.8, edgecolor='white', linewidth=0.5),
               color='white', fontweight='bold', zorder=11)
        
        # Add direction indicator (small line)
        dir_length = 0.02
        direction_end = drone.pos + drone.direction * dir_length
        ax.plot([drone.pos[0], direction_end[0]], 
                [drone.pos[1], direction_end[1]], 
                'w-', lw=1.5, alpha=0.6, zorder=9)
    
    # Customize plot
    ax.set_aspect('equal')
    ax.set_title('UAV Swarm Trust Simulation', color='white', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('X Position', color='white', fontsize=12)
    ax.set_ylabel('Y Position', color='white', fontsize=12)
    
    # Customize ticks
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white', linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=COLORS['trusted'], edgecolor='darkgreen', label='Trusted Drone'),
        Patch(facecolor=COLORS['malicious'], edgecolor='darkred', label='Malicious Drone'),
        Patch(facecolor=COLORS['isolated'], edgecolor='red', label='Isolated Drone'),
        Patch(facecolor=COLORS['enemy'], edgecolor='darkorange', label='Enemy Signal'),
        Patch(facecolor=COLORS['unknown'], edgecolor='gold', label='Unknown Signal'),
        Line2D([0], [0], color='cyan', lw=2, label='Trusted Communication'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Blocked/Malicious'),
        Line2D([0], [0], color='blue', lw=0, alpha=0.1, label=f'Comm Range ({COMM_RANGE})')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left',
             facecolor='black', edgecolor='white',
             labelcolor='white', fontsize=8)
    
    # Add info text
    info_text = f'Swarm Size: {len(st.session_state.drones)}/{MAX_DRONES} | '
    info_text += f'Active Comms: {len(st.session_state.communications)} | '
    info_text += f'Malicious: {len(st.session_state.malicious_drones)}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, color='white', ha='left', va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    plt.tight_layout()
    return fig

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(
    page_title="UAV Swarm Trust Monitor",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .log-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        height: 700px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #e0e0e0;
        border: 1px solid #2d3748;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .drone-card {
        background: rgba(45, 55, 72, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #48bb78;
        transition: all 0.3s ease;
    }
    
    .drone-card:hover {
        transform: translateX(5px);
        background: rgba(56, 68, 88, 0.9);
    }
    
    .malicious-drone-card {
        border-left: 4px solid #f56565 !important;
        background: rgba(155, 44, 44, 0.2) !important;
    }
    
    .isolated-drone-card {
        border-left: 4px solid #ed8936 !important;
        background: rgba(197, 121, 40, 0.2) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #a0aec0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a5568 !important;
        color: white !important;
    }
    
    /* Custom scrollbar */
    .log-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .log-container::-webkit-scrollbar-track {
        background: #1a202c;
        border-radius: 4px;
    }
    
    .log-container::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }
    
    .log-container::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }
</style>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1.2, 2])

# =======================
# LEFT PANEL - LOGS AND INFO
# =======================
with col1:
    st.title("📡 UAV Trust Monitor")
    
    # Metrics row
    st.markdown("### 📊 Swarm Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    trusted_count = sum(1 for d in st.session_state.drones if d.trusted and not d.isolated)
    malicious_count = sum(1 for d in st.session_state.drones if not d.trusted)
    isolated_count = sum(1 for d in st.session_state.drones if d.isolated)
    active_comms = len(st.session_state.communications)
    
    with metric_col1:
        st.metric("Total Drones", len(st.session_state.drones), 
                 delta=f"Max: {MAX_DRONES}" if len(st.session_state.drones) < MAX_DRONES else "MAX")
    with metric_col2:
        st.metric("Trusted", trusted_count)
    with metric_col3:
        st.metric("Malicious", malicious_count, 
                 delta="detected" if malicious_count > 0 else "none")
    with metric_col4:
        st.metric("Active Comms", active_comms)
    
    # Control buttons
    st.markdown("---")
    st.markdown("### 🎮 Simulation Controls")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("⏸️ Pause" if st.session_state.simulation_running else "▶️ Resume", 
                    key="pause_btn"):
            st.session_state.simulation_running = not st.session_state.simulation_running
            action = "paused" if not st.session_state.simulation_running else "resumed"
            add_log(f"Simulation {action}")
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Reset Simulation", key="reset_btn"):
            st.session_state.drones = create_initial_drones()
            st.session_state.logs = []
            st.session_state.communications = []
            st.session_state.malicious_drones = set()
            st.session_state.selected_drone = None
            st.session_state.simulation_running = True
            open(LOG_FILE_PATH, "w").close()  # Clear log file

            add_log("Simulation reset to initial state")
            st.rerun()
    
    # Drone list
    st.markdown("---")
    st.markdown("### 🛸 Active Drones")
    
    for drone in st.session_state.drones:
        # Determine status and emoji
        if drone.isolated:
            status = "🔴 ISOLATED"
            emoji = "🔴"
            card_class = "isolated-drone-card"
        elif not drone.trusted:
            status = "⚠️ MALICIOUS"
            emoji = "⚠️"
            card_class = "malicious-drone-card"
        else:
            status = "✅ TRUSTED"
            emoji = "✅"
            card_class = "drone-card"
        
        # Create a clickable drone card
        with st.container():
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"**{emoji} {drone.label}**")
                st.caption(f"Model: {drone.model_name}")
                if drone.last_verification_distance:
                    st.caption(f"Last score: {drone.last_verification_distance:.3f}")
            
            with col_right:
                if st.button("📊", key=f"view_{drone.id}", help=f"View details of {drone.label}"):
                    st.session_state.selected_drone = drone.id
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Log display
    st.markdown("---")
    st.markdown("### 📋 Activity Log")
    
    # Create a container for logs with custom styling
    log_container = st.container(height=400)
    
    with log_container:
        # Display logs in reverse order (newest first)
        for log in reversed(st.session_state.logs[-20:]):
            # Color code log messages
            if "MALICIOUS" in log:
                st.error(log, icon="🚨")
            elif "WARNING" in log or "BLOCKED" in log:
                st.warning(log, icon="⚠️")
            elif "COMM:" in log:
                st.info(log, icon="📡")
            elif "ADDED" in log:
                st.success(log, icon="🆕")
            else:
                st.write(log)

# =======================
# RIGHT PANEL - SWARM VISUALIZATION
# =======================
with col2:
    # Header with swarm size
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("🛰️ Swarm Simulation")
        st.caption("Drones move autonomously and communicate when in range. Malicious drones are isolated.")
    
    with header_col2:
        # Display swarm size in a prominent way
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); 
                    border-radius: 15px; 
                    padding: 20px; 
                    text-align: center;
                    border: 2px solid #48bb78;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; color: #a0aec0;">Swarm Size</h3>
            <h1 style="margin: 10px 0; color: white; font-size: 48px;">{len(st.session_state.drones)}</h1>
            <p style="margin: 0; color: #a0aec0;">/ {MAX_DRONES} max</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Update simulation
    update_simulation()
    
    # Create and display swarm visualization
    swarm_fig = create_swarm_visualization()
    st.pyplot(swarm_fig, use_container_width=True)
    plt.close(swarm_fig)
    
    # Add drone button at bottom right
    st.markdown("---")
    
    # Create a centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if len(st.session_state.drones) < MAX_DRONES:
            if st.button("➕ Add New Drone to Swarm", use_container_width=True, 
                        type="primary", help="Add a random drone to the simulation"):
                model_name, npz_path = pick_random_drone()
                if model_name and npz_path:
                    new_id = f"D{len(st.session_state.drones) + 1}"
                    new_drone = Drone(
                        drone_id=new_id,
                        label=f"{model_name}_{len(st.session_state.drones) + 1}",
                        model_name=model_name,
                        npz_path=npz_path,
                        model_instance=model
                    )
                    st.session_state.drones.append(new_drone)
                    add_log(f"ADDED: New drone {new_drone.label} ({model_name})")
                    simulate_baselines(st.session_state.base_station, len(st.session_state.drones))
                    st.rerun()
        else:
            st.warning(f"⚠️ Maximum drone limit reached ({MAX_DRONES})", icon="⚠️")

# =======================
# DETAILED DRONE INFORMATION (Expands when drone is selected)
# =======================
if st.session_state.selected_drone:
    selected_drone = next((d for d in st.session_state.drones 
                          if d.id == st.session_state.selected_drone), None)
    
    if selected_drone:
        st.markdown("---")
        
        # Create a nice header for drone details
        status_color = "#48bb78" if selected_drone.trusted else "#f56565"
        status_text = "TRUSTED ✅" if selected_drone.trusted else "MALICIOUS 🚨"
        if selected_drone.isolated:
            status_text = "ISOLATED 🔴"
            status_color = "#ed8936"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-left: 6px solid {status_color};
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
            <h2 style="margin: 0; color: white;">📊 Drone Details: {selected_drone.label}</h2>
            <h3 style="margin: 10px 0; color: {status_color};">Status: {status_text}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different information
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Status Info", "📡 RF Analysis", "🎯 Verification", "📜 History"])
        
        with tab1:
            # Status information in a grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🆔 Basic Info")
                st.write(f"**ID:** {selected_drone.id}")
                st.write(f"**Model:** {selected_drone.model_name}")
                st.write(f"**Position:** ({selected_drone.pos[0]:.3f}, {selected_drone.pos[1]:.3f})")
                st.write(f"**Speed:** {selected_drone.speed:.4f}")
            
            with col2:
                st.markdown("#### 🛡️ Security Status")
                st.write(f"**Trusted:** {'✅ Yes' if selected_drone.trusted else '❌ No'}")
                st.write(f"**Isolated:** {'✅ Yes' if selected_drone.isolated else '❌ No'}")
                st.write(f"**Comm Range:** {COMM_RANGE:.2f}")
                if selected_drone.last_verification_distance:
                    st.write(f"**Last Score:** {selected_drone.last_verification_distance:.3f}")
            
            with col3:
                st.markdown("#### 🎯 Direction")
                # Display direction as arrow and coordinates
                dir_x, dir_y = selected_drone.direction
                angle = np.degrees(np.arctan2(dir_y, dir_x))
                
                st.write(f"**Direction:** ({dir_x:.3f}, {dir_y:.3f})")
                st.write(f"**Angle:** {angle:.1f}°")
                
                # Small direction visualization
                fig_dir, ax_dir = plt.subplots(figsize=(3, 3))
                ax_dir.arrow(0, 0, dir_x*0.8, dir_y*0.8, head_width=0.1, head_length=0.1, 
                           fc=status_color, ec=status_color, linewidth=2)
                ax_dir.set_xlim(-1, 1)
                ax_dir.set_ylim(-1, 1)
                ax_dir.set_aspect('equal')
                ax_dir.axis('off')
                ax_dir.set_facecolor('#1a1a2e')
                fig_dir.patch.set_facecolor('#1a1a2e')
                st.pyplot(fig_dir, use_container_width=True)
                plt.close(fig_dir)
        
        with tab2:
            # RF Signal Analysis
            st.markdown("#### 📶 RF Signal Analysis")
            
            if selected_drone.iq_data is not None:
                # IQ Data plots
                iq_fig = plot_iq_data(selected_drone.iq_data, selected_drone.label)
                st.pyplot(iq_fig)
                plt.close(iq_fig)
                
                # Spectrogram plots
                if selected_drone.spectrogram_data is not None:
                    spec_fig = plot_spectrogram(selected_drone.spectrogram_data, selected_drone.label)
                    st.pyplot(spec_fig)
                    plt.close(spec_fig)
                else:
                    st.info("No spectrogram data available for this drone")
            else:
                st.warning("No IQ data available for this drone")
        
        with tab3:
            # Verification results
            st.markdown("#### 🎯 Last Verification Result")
            
            if selected_drone.last_verification_result is not None:
                # Create a nice result display
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Big result indicator
                    result_text = "✅ TRUSTED" if selected_drone.last_verification_result else "❌ MALICIOUS"
                    result_color = "#48bb78" if selected_drone.last_verification_result else "#f56565"
                    
                    st.markdown(f"""
                    <div style="background: {result_color};
                                border-radius: 15px;
                                padding: 30px;
                                text-align: center;
                                margin: 10px 0;">
                        <h1 style="color: white; margin: 0;">{result_text}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    # Score and threshold information
                    score = selected_drone.last_verification_distance
                    
                    # Create a progress bar for the score
                    progress_value = min(score / (THRESHOLD * 1.5), 1.0)
                    
                    st.metric("Distance Score", f"{score:.3f}",
                             delta="Below Threshold ✅" if score < THRESHOLD else "Above Threshold ⚠️",
                             delta_color="normal" if score < THRESHOLD else "off")
                    
                    st.progress(progress_value, 
                               text=f"Score: {score:.3f} | Threshold: {THRESHOLD}")
                    
                    # Interpretation
                    if score < THRESHOLD:
                        st.success(f"✅ This drone is trusted. The distance score ({score:.3f}) is below the threshold ({THRESHOLD}).")
                    else:
                        st.error(f"⚠️ This drone is not trusted. The distance score ({score:.3f}) exceeds the threshold ({THRESHOLD}).")
                        
                    # Score comparison
                    st.write("**Score Interpretation:**")
                    if score < 0.5:
                        st.write("• Very similar signal (high confidence)")
                    elif score < 1.0:
                        st.write("• Similar signal (moderate confidence)")
                    elif score < 1.5:
                        st.write("• Different signal (low confidence)")
                    else:
                        st.write("• Very different signal (highly suspicious)")
            else:
                st.info("No verification has been performed yet for this drone.")
        
        with tab4:
            # Communication history
            st.markdown("#### 📜 Communication History")
            
            if selected_drone.communication_history:
                for i, comm in enumerate(reversed(selected_drone.communication_history[-10:]), 1):
                    timestamp = time.strftime("%H:%M:%S")
                    st.write(f"{i}. [{timestamp}] {comm}")
            else:
                st.info("No communication history recorded yet.")
            
            # Recent logs involving this drone
            st.markdown("#### 📋 Recent Activity")
            drone_logs = [log for log in st.session_state.logs[-15:] 
                         if selected_drone.label in log]
            
            if drone_logs:
                for log in reversed(drone_logs):
                    st.write(log)
            else:
                st.info("No recent activity involving this drone.")



# =======================
# PHASE-2 EVALUATION & COMPARISON (F)
# =======================

st.markdown("---")
st.markdown("## 🔬 Phase-2 Security & Communication Comparison")


def classical_latency(swarm_size):
    # TLS + cert + symmetric key
    return 4 + 0.4 * swarm_size   # ms

def classical_bandwidth(swarm_size):
    return 1.2 + 0.15 * swarm_size  # KB

def quantum_latency(swarm_size):
    # Ideal quantum link (upper bound)
    return 10 + 0.8 * swarm_size  # ms

def quantum_bandwidth(swarm_size):
    return 4 + 0.7 * swarm_size   # KB


def plot_latency_comparison_lines():
    hybrid = st.session_state.base_station.metrics["hybrid"]

    if len(hybrid) < 2:
        st.info("Waiting for more PQC sessions to plot latency trends...")
        return None

    df = pd.DataFrame(hybrid)

    # Unique swarm sizes encountered
    swarm_sizes = sorted(df["swarm_size"].unique())

    # Hybrid = actual measured average
    hybrid_latency = [
        df[df["swarm_size"] == s]["latency"].mean()
        for s in swarm_sizes
    ]

    # Baselines
    classical_latency_vals = [classical_latency(s) for s in swarm_sizes]
    quantum_latency_vals = [quantum_latency(s) for s in swarm_sizes]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(swarm_sizes, classical_latency_vals,
            marker="^", linestyle="--", label="Classical")

    ax.plot(swarm_sizes, hybrid_latency,
            marker="o", linewidth=2.5, label="Hybrid PQC")

    ax.plot(swarm_sizes, quantum_latency_vals,
            marker="s", linestyle=":", label="Fully Quantum")

    ax.set_xlabel("Swarm Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Scaling vs Swarm Size")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def dynamic_phase2_plot():
    bs = st.session_state.base_station

    def avg(metric, key):
        if not metric:
            return np.nan
        return sum(m[key] for m in metric) / len(metric)

    data = {
        "Method": ["Classical", "Hybrid PQC", "Fully Quantum"],
        "Latency (ms)": [
            avg(bs.metrics["classical"], "latency"),
            avg(bs.metrics["hybrid"], "latency"),
            avg(bs.metrics["quantum"], "latency")
        ],
        "Bandwidth (KB)": [
            avg(bs.metrics["classical"], "bandwidth"),
            avg(bs.metrics["hybrid"], "bandwidth"),
            avg(bs.metrics["quantum"], "bandwidth")
        ],
        "Security Score": [
            avg(bs.metrics["classical"], "security"),
            avg(bs.metrics["hybrid"], "security"),
            avg(bs.metrics["quantum"], "security")
        ]
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 4))
    df.set_index("Method").plot(kind="bar", ax=ax)
    ax.set_title(f"PQC Metrics vs Swarm Size = {len(st.session_state.drones)}")
    ax.grid(True, alpha=0.3)

    return fig

def plot_dynamic_line_metrics():
    bs = st.session_state.base_station
    hybrid = bs.metrics["hybrid"]

    if len(hybrid) < 2:
        st.info("Waiting for more PQC sessions to plot trends...")
        return None

    df = pd.DataFrame(hybrid)

    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Latency line
    ax1.plot(
        df["swarm_size"],
        df["latency"],
        marker="o",
        label="Hybrid PQC Latency (ms)"
    )
    ax1.set_xlabel("Swarm Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, alpha=0.3)

    # Second Y-axis for bandwidth
    ax2 = ax1.twinx()
    ax2.plot(
        df["swarm_size"],
        df["bandwidth"],
        marker="s",
        color="orange",
        label="Hybrid PQC Bandwidth (KB)"
    )
    ax2.set_ylabel("Bandwidth (KB)")

    # Legend handling
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Hybrid PQC Scalability vs Swarm Size")

    return fig


if plot_dynamic_line_metrics():
    st.pyplot(plot_dynamic_line_metrics(), use_container_width=True)



if not st.session_state.base_station.metrics["hybrid"]:
    st.info("Waiting for PQC sessions to establish...")
else:
    st.pyplot(dynamic_phase2_plot(), use_container_width=True)



# Auto-refresh when simulation is running
if st.session_state.simulation_running:
    # Small delay to control refresh rate
    time.sleep(0.05)
    st.rerun()