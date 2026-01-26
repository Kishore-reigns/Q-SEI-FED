import streamlit as st
import numpy as np
import torch
import os
import random
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
import matplotlib
import io
from scipy import signal
import pandas as pd
from datetime import datetime
import json
import glob
from collections import defaultdict, deque, Counter
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use Agg backend
matplotlib.use("Agg")

# =======================
# CONFIGURATION
# =======================
DATASET_ROOT = r"F:\K DRIVE\MIT_Learnings\Sem8\dataset\code\sei_dataset4"
LOG_FILE_PATH = "uav_simulation_logs.txt"

# Verify dataset exists
if not os.path.exists(DATASET_ROOT):
    st.error(f"Dataset path not found: {DATASET_ROOT}")
    st.stop()

MAX_DRONES = 20
INITIAL_DRONES = 6
WORLD_SIZE = 1.0
COMM_RANGE = 0.35
THRESHOLD = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPDATE_INTERVAL = 0.05

# Byzantine consensus parameters
MIN_CONSENSUS_RATIO = 0.75
TRUST_DECAY_RATE = 0.97
MIN_TRUST_SCORE = 0.4
MAX_HISTORY_SIZE = 100

# SEI matching thresholds
SEI_MATCH_THRESHOLD = 0.85
CFO_THRESHOLD = 1000
IQ_IMBALANCE_THRESHOLD = 0.05

# Colors for visualization
COLORS = {
    'trusted': '#00FF00',
    'malicious': '#FF0000',
    'communication': '#4169E1',
    'isolated': '#FFA500',
    'background': '#0E1117',
    'text': '#FFFFFF',
    'enemy': '#FF6B6B',
    'unknown': '#FFD700',
    'poisoned': '#FF00FF',
    'gbs': '#9400D3',
    'suspicious': '#FFA500',
    'byzantine': '#8B0000'
}

# =======================
# DATASET LOADING WITH SEI PARAMETERS
# =======================
@st.cache_resource
def load_dataset_samples():
    """Load samples from the dataset directory with SEI parameters"""
    samples_by_type = defaultdict(list)
    
    # Check for subdirectories (drone types)
    drone_types = [d for d in os.listdir(DATASET_ROOT) 
                  if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    if not drone_types:
        # Look for NPZ files directly
        npz_files = glob.glob(os.path.join(DATASET_ROOT, "*.npz"))
        if npz_files:
            for npz_file in npz_files:
                extract_sample(npz_file, samples_by_type)
    else:
        # Load from subdirectories
        for drone_type in drone_types:
            drone_path = os.path.join(DATASET_ROOT, drone_type)
            npz_files = glob.glob(os.path.join(drone_path, "*.npz"))
            for npz_file in npz_files:
                extract_sample(npz_file, samples_by_type, drone_type)
    
    # If still no samples, create simulated ones
    if not samples_by_type:
        samples_by_type = create_simulated_samples()
    
    return samples_by_type

def extract_sample(npz_file, samples_by_type, drone_type=None):
    """Extract sample from NPZ file"""
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Determine drone type
        if drone_type is None:
            # Try to extract from filename or data
            if 'manufacturer' in data:
                drone_type = str(data['manufacturer'].item())
            else:
                # Extract from filename
                filename = os.path.basename(npz_file)
                for known_type in ['DJI', 'FUTABA', 'ENEMY', 'GRAUPNER', 'TURNIGY', 'UNKNOWN']:
                    if known_type in filename.upper():
                        drone_type = known_type
                        break
                if drone_type is None:
                    drone_type = 'UNKNOWN'
        
        # Extract SEI parameters with defaults
        sei_params = {}
        if 'sei_params' in data:
            sei_data = data['sei_params'].item()
            if isinstance(sei_data, dict):
                sei_params = sei_data
        else:
            # Generate realistic SEI parameters based on drone type
            sei_params = generate_sei_params(drone_type)
        
        # Extract IQ data
        iq_data = None
        if 'iq' in data:
            iq_data = data['iq']
        elif 'signal' in data:
            iq_data = data['signal']
        
        # Extract spectrogram
        spectrogram = None
        if 'spectrogram' in data:
            spectrogram = data['spectrogram']
        
        sample = {
            'file_path': npz_file,
            'drone_type': drone_type,
            'iq_data': iq_data,
            'spectrogram': spectrogram,
            'snr_db': float(data['snr_db']) if 'snr_db' in data else random.uniform(15, 30),
            'instance_id': str(data['instance_id'].item()) if 'instance_id' in data else 
                          f"{drone_type}_{hash(npz_file) % 1000:03d}",
            'manufacturer': str(data['manufacturer'].item()) if 'manufacturer' in data else drone_type,
            'is_known': bool(data['is_known'].item()) if 'is_known' in data else True,
            'sei_params': sei_params
        }
        
        # Generate IQ data if missing
        if sample['iq_data'] is None:
            sample['iq_data'] = generate_iq_from_sei(sei_params, sample['snr_db'])
        
        # Generate spectrogram if missing
        if sample['spectrogram'] is None:
            sample['spectrogram'] = generate_spectrogram(sample['iq_data'])
        
        samples_by_type[drone_type].append(sample)
        
    except Exception as e:
        # Skip problematic files
        pass

def create_simulated_samples():
    """Create simulated samples when real data is not available"""
    samples_by_type = defaultdict(list)
    
    drone_types = ['DJI_001', 'DJI_002', 'DJI_003', 'ENEMY_DJI', 'ENEMY_FUTABA', 
                   'FUTABA_T7_001', 'FUTABA_T14_001', 'GRAUPNER_001', 
                   'TURNIGY_001', 'UNKNOWN_NOISE']
    
    for drone_type in drone_types:
        for i in range(3):  # Create 3 samples per type
            sei_params = generate_sei_params(drone_type)
            snr_db = random.uniform(15, 30)
            
            sample = {
                'file_path': f"simulated/{drone_type}_{i}.npz",
                'drone_type': drone_type,
                'iq_data': generate_iq_from_sei(sei_params, snr_db),
                'spectrogram': None,
                'snr_db': snr_db,
                'instance_id': f"{drone_type}_{i:03d}",
                'manufacturer': drone_type.split('_')[0],
                'is_known': 'UNKNOWN' not in drone_type,
                'sei_params': sei_params
            }
            sample['spectrogram'] = generate_spectrogram(sample['iq_data'])
            
            samples_by_type[drone_type].append(sample)
    
    return samples_by_type

def generate_sei_params(drone_type):
    """Generate realistic SEI parameters based on drone type"""
    params = {
        'cfo_hz': random.uniform(-500, 500),
        'iq_gain': random.uniform(0.95, 1.05),
        'iq_phase': random.uniform(-5, 5),
        'pa_alpha': random.uniform(2.0, 3.0),
        'phase_noise': random.uniform(1, 10),
        'filter_fc': random.uniform(0.8, 1.2),
    }
    
    # Add manufacturer-specific biases
    if 'DJI' in drone_type:
        params['cfo_hz'] = random.uniform(-200, 200)
        params['iq_gain'] = random.uniform(0.98, 1.02)
    elif 'FUTABA' in drone_type:
        params['phase_noise'] = random.uniform(5, 15)
    elif 'ENEMY' in drone_type:
        params['cfo_hz'] = random.uniform(-1000, 1000)
        params['iq_gain'] = random.uniform(0.9, 1.1)
    elif 'UNKNOWN' in drone_type:
        params['cfo_hz'] = random.uniform(-800, 800)
        params['phase_noise'] = random.uniform(10, 20)
    
    return params

def generate_iq_from_sei(sei_params, snr_db):
    """Generate IQ data from SEI parameters"""
    n_samples = 1024
    t = np.linspace(0, 1, n_samples)
    
    # Base signal
    freq = 30
    i_signal = np.sin(2 * np.pi * freq * t)
    q_signal = np.cos(2 * np.pi * freq * t)
    
    # Apply SEI distortions
    # Carrier frequency offset
    cfo_phase = 2 * np.pi * sei_params['cfo_hz'] * t / 1000
    i_signal = i_signal * np.cos(cfo_phase) - q_signal * np.sin(cfo_phase)
    q_signal = i_signal * np.sin(cfo_phase) + q_signal * np.cos(cfo_phase)
    
    # I/Q imbalance
    i_signal = i_signal * sei_params['iq_gain']
    q_phase_rad = np.deg2rad(sei_params['iq_phase'])
    q_signal = q_signal * np.cos(q_phase_rad)
    
    # Power amplifier nonlinearity
    alpha = sei_params['pa_alpha']
    i_signal = i_signal - alpha * i_signal**3 / 3
    q_signal = q_signal - alpha * q_signal**3 / 3
    
    # Add phase noise
    phase_noise = np.deg2rad(sei_params['phase_noise']) * np.random.randn(n_samples)
    i_noisy = i_signal * np.cos(phase_noise) - q_signal * np.sin(phase_noise)
    q_noisy = i_signal * np.sin(phase_noise) + q_signal * np.cos(phase_noise)
    
    # Add AWGN based on SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = 1 / snr_linear
    i_noisy += np.random.randn(n_samples) * np.sqrt(noise_power)
    q_noisy += np.random.randn(n_samples) * np.sqrt(noise_power)
    
    # Apply filter
    if sei_params['filter_fc'] != 1.0:
        b, a = signal.butter(3, sei_params['filter_fc'])
        i_noisy = signal.filtfilt(b, a, i_noisy)
        q_noisy = signal.filtfilt(b, a, q_noisy)
    
    return i_noisy + 1j * q_noisy

def generate_spectrogram(iq_data):
    """Generate spectrogram from IQ data"""
    f, t, Sxx = signal.spectrogram(iq_data, fs=1000, nperseg=64, noverlap=32)
    return 10 * np.log10(np.abs(Sxx) + 1e-10)

# =======================
# BYZANTINE-RESISTANT TRUST MANAGEMENT
# =======================
class ByzantineResistantTrustManager:
    """Implements Byzantine fault-tolerant trust management"""
    
    def __init__(self):
        self.trust_scores = defaultdict(lambda: 0.7)
        self.trust_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_SIZE))
        self.consensus_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_SIZE))
        self.sei_fingerprints = {}
        self.update_vectors = {}
        
    def calculate_sei_similarity(self, sei_params1, sei_params2):
        """Calculate similarity between two SEI parameter sets"""
        if not sei_params1 or not sei_params2:
            return 0.0
        
        similarities = []
        
        # Carrier frequency offset
        cfo_diff = abs(sei_params1.get('cfo_hz', 0) - sei_params2.get('cfo_hz', 0))
        cfo_sim = max(0, 1 - cfo_diff / CFO_THRESHOLD)
        similarities.append(cfo_sim)
        
        # I/Q imbalance
        iq_gain_diff = abs(sei_params1.get('iq_gain', 1) - sei_params2.get('iq_gain', 1))
        iq_phase_diff = abs(sei_params1.get('iq_phase', 0) - sei_params2.get('iq_phase', 0))
        iq_sim = max(0, 1 - (iq_gain_diff + iq_phase_diff/10) / IQ_IMBALANCE_THRESHOLD)
        similarities.append(iq_sim)
        
        # Other parameters
        for key in ['pa_alpha', 'phase_noise', 'filter_fc']:
            if key in sei_params1 and key in sei_params2:
                val1 = sei_params1[key]
                val2 = sei_params2[key]
                if key == 'filter_fc':
                    diff = abs(val1 - val2) / 0.4
                else:
                    diff = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)
                similarities.append(max(0, 1 - diff))
        
        return np.mean(similarities) if similarities else 0.0
    
    def verify_drone_identity(self, drone_id, current_sei, claimed_id):
        """Verify drone identity using SEI fingerprinting"""
        if drone_id not in self.sei_fingerprints:
            # First time seeing this drone, store fingerprint
            self.sei_fingerprints[drone_id] = {
                'sei_params': current_sei,
                'first_seen': time.time(),
                'verification_count': 0
            }
            return True, 1.0
        
        stored_fingerprint = self.sei_fingerprints[drone_id]['sei_params']
        similarity = self.calculate_sei_similarity(stored_fingerprint, current_sei)
        
        self.sei_fingerprints[drone_id]['verification_count'] += 1
        
        if similarity >= SEI_MATCH_THRESHOLD:
            # Update fingerprint with moving average
            for key in stored_fingerprint:
                if key in current_sei:
                    alpha = 0.1
                    stored_fingerprint[key] = (1 - alpha) * stored_fingerprint[key] + alpha * current_sei[key]
            return True, similarity
        else:
            return False, similarity
    
    def byzantine_consensus_check(self, drone_id, update_vector, all_updates):
        """Check if an update is Byzantine using consensus"""
        if len(all_updates) < 2:
            return True, 1.0
        
        # Convert updates to numpy arrays
        update_arrays = []
        for update in all_updates:
            if isinstance(update, dict):
                vec = update.get('gradients', update.get('weights', []))
                if isinstance(vec, list):
                    update_arrays.append(np.array(vec))
            elif isinstance(update, np.ndarray):
                update_arrays.append(update)
        
        if len(update_arrays) < 2:
            return True, 1.0
        
        # Calculate median update
        stacked = np.stack(update_arrays)
        median_update = np.median(stacked, axis=0)
        
        # Calculate cosine similarity to median
        if isinstance(update_vector, dict):
            test_vec = np.array(update_vector.get('gradients', update_vector.get('weights', [])))
        else:
            test_vec = np.array(update_vector)
        
        if len(test_vec) == 0 or len(median_update) == 0 or len(test_vec) != len(median_update):
            return False, 0.0
        
        # Normalize vectors
        test_vec_norm = test_vec / (np.linalg.norm(test_vec) + 1e-10)
        median_norm = median_update / (np.linalg.norm(median_update) + 1e-10)
        
        similarity = np.dot(test_vec_norm, median_norm)
        
        # Check if enough drones agree
        consensus_scores = []
        for vec in update_arrays:
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            score = np.dot(vec_norm, median_norm)
            consensus_scores.append(score)
        
        consensus_ratio = sum(s > 0.7 for s in consensus_scores) / len(consensus_scores)
        
        return similarity > 0.7 and consensus_ratio >= MIN_CONSENSUS_RATIO, similarity
    
    def update_trust_score(self, drone_id, verification_result, sei_similarity, 
                          consensus_result, time_factor=1.0):
        """Update trust score with multiple factors"""
        current_score = self.trust_scores[drone_id]
        
        # Base score from verification
        if verification_result:
            base_score = 0.8 + 0.2 * sei_similarity
        else:
            base_score = 0.2 * sei_similarity
        
        # Consensus factor
        consensus_factor = 1.0 if consensus_result else 0.3
        
        # Time decay
        time_decay = TRUST_DECAY_RATE ** time_factor
        
        # Calculate new score
        new_score = (current_score * 0.4 + base_score * 0.4 + consensus_factor * 0.2) * time_decay
        
        # Ensure score is within bounds
        new_score = max(MIN_TRUST_SCORE, min(1.0, new_score))
        
        # Store in history
        self.trust_scores[drone_id] = new_score
        self.trust_history[drone_id].append({
            'timestamp': time.time(),
            'score': new_score,
            'verification': verification_result,
            'sei_similarity': sei_similarity,
            'consensus': consensus_result
        })
        
        return new_score
    
    def get_trust_level(self, drone_id):
        """Get trust level categorization"""
        score = self.trust_scores[drone_id]
        if score >= 0.8:
            return "TRUSTED"
        elif score >= 0.6:
            return "SUSPICIOUS"
        elif score >= 0.4:
            return "MALICIOUS"
        else:
            return "BYZANTINE"
    
    def detect_anomalies(self, drone_id, current_behavior):
        """Detect behavioral anomalies"""
        history = list(self.trust_history[drone_id])
        if len(history) < 5:
            return False, 0.0
        
        # Check for sudden trust drop
        recent_scores = [h['score'] for h in history[-5:]]
        if len(recent_scores) >= 3:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            if trend < -0.1:
                return True, abs(trend)
        
        # Check SEI consistency
        recent_sei = [h.get('sei_similarity', 1.0) for h in history[-5:] if 'sei_similarity' in h]
        if recent_sei and np.mean(recent_sei) < 0.7:
            return True, 1 - np.mean(recent_sei)
        
        return False, 0.0

# =======================
# ADVANCED DRONE CLASS WITH SEI
# =======================
class AdvancedDrone:
    """Drone class with SEI capabilities and attack resistance"""
    
    def __init__(self, drone_id, drone_type, sample_data, is_trusted=True, is_malicious=False):
        self.id = drone_id
        self.drone_type = drone_type
        self.sample_data = sample_data
        self.trusted = is_trusted
        self.is_malicious = is_malicious
        self.isolated = False
        self.poisoned = False
        self.compromised = False
        
        # Add label attribute
        self.label = f"{drone_type}_{drone_id}"
        
        # Extract SEI parameters
        self.sei_params = sample_data.get('sei_params', generate_sei_params(drone_type))
        self.sei_fingerprint = self.calculate_sei_fingerprint()
        
        # Position and movement
        self.pos = np.array([random.uniform(0.1, WORLD_SIZE-0.1), 
                            random.uniform(0.1, WORLD_SIZE-0.1)])
        
        # Movement parameters
        self.speed = self.get_speed_from_type()
        self.direction = self.initialize_direction()
        
        # Data for visualization
        self.iq_data = sample_data.get('iq_data', generate_iq_from_sei(self.sei_params, 20))
        self.spectrogram_data = sample_data.get('spectrogram', 
                                               generate_spectrogram(self.iq_data))
        
        # Trust and verification
        self.trust_score = 0.8 if is_trusted else 0.3
        self.verification_history = deque(maxlen=10)
        self.communication_history = deque(maxlen=20)
        
        # Model updates
        self.model_updates = []
        self.last_update_time = 0
        self.update_interval = random.uniform(3, 8)
        self.update_history = deque(maxlen=10)
        
        # Attack parameters
        self.attack_type = None
        self.attack_strength = 0.0
        if is_malicious:
            self.initialize_attack_parameters()
    
    def get_speed_from_type(self):
        """Get speed based on drone type"""
        speed_map = {
            'DJI': 0.006,
            'FUTABA': 0.005,
            'GRAUPNER': 0.004,
            'TURNIGY': 0.005,
            'ENEMY': 0.008,
            'UNKNOWN': 0.004
        }
        
        for key in speed_map:
            if key in self.drone_type.upper():
                return speed_map[key]
        return 0.005
    
    def initialize_direction(self):
        """Initialize movement direction"""
        center = np.array([WORLD_SIZE/2, WORLD_SIZE/2])
        direction = center - self.pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Add randomness
        direction += np.random.randn(2) * 0.2
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def calculate_sei_fingerprint(self):
        """Calculate unique SEI fingerprint"""
        params = self.sei_params
        fingerprint_str = f"{params.get('cfo_hz',0):.2f}_{params.get('iq_gain',1):.4f}_" \
                         f"{params.get('iq_phase',0):.2f}_{params.get('pa_alpha',2.5):.3f}"
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def initialize_attack_parameters(self):
        """Initialize attack parameters for malicious drones"""
        attack_types = ['MODEL_POISONING', 'SYBIL', 'REPLAY', 'EVASIVE']
        self.attack_type = random.choice(attack_types)
        self.attack_strength = random.uniform(0.5, 1.0)
        
        # Adjust SEI parameters for attack
        if self.attack_type == 'REPLAY':
            self.sei_params['cfo_hz'] *= random.uniform(0.9, 1.1)
            self.sei_params['iq_gain'] *= random.uniform(0.95, 1.05)
        elif self.attack_type == 'EVASIVE':
            self.sei_params['cfo_hz'] += random.uniform(-200, 200)
            self.sei_params['phase_noise'] *= random.uniform(1.5, 3.0)
    
    def move(self):
        """Advanced movement with attack behaviors"""
        if self.isolated:
            # Isolated drones move erratically
            self.direction += np.random.randn(2) * 0.3
            speed = self.speed * 0.2
        elif self.is_malicious:
            # Malicious drones have strategic movement
            if self.attack_type == 'EVASIVE':
                self.direction += np.random.randn(2) * 0.4
                speed = self.speed * 1.2
            else:
                # Move toward trusted drones
                trusted_positions = self.find_trusted_drones()
                if trusted_positions:
                    target = random.choice(trusted_positions)
                    self.direction = target - self.pos
                    self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
                    self.direction += np.random.randn(2) * 0.1
                    speed = self.speed * 1.1
                else:
                    self.direction += np.random.randn(2) * 0.2
                    speed = self.speed
        else:
            # Normal swarm movement
            self.direction += np.random.randn(2) * 0.1
            speed = self.speed
        
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        
        # Update position
        self.pos += self.direction * speed
        
        # Boundary handling
        self.handle_boundaries()
        
        # Update trust score decay
        if not self.is_malicious:
            self.trust_score *= TRUST_DECAY_RATE
    
    def find_trusted_drones(self):
        """Find positions of trusted drones (for malicious drones)"""
        if 'drones' not in st.session_state:
            return []
        
        trusted_positions = []
        for drone in st.session_state.drones:
            if drone.trusted and not drone.isolated and drone.id != self.id:
                distance = euclidean(self.pos, drone.pos)
                if distance < COMM_RANGE * 2:
                    trusted_positions.append(drone.pos)
        
        return trusted_positions
    
    def handle_boundaries(self):
        """Handle boundary collisions"""
        bounce_factor = 0.7
        
        if self.pos[0] < 0.02:
            self.pos[0] = 0.02
            self.direction[0] = abs(self.direction[0]) * bounce_factor
        elif self.pos[0] > WORLD_SIZE - 0.02:
            self.pos[0] = WORLD_SIZE - 0.02
            self.direction[0] = -abs(self.direction[0]) * bounce_factor
        
        if self.pos[1] < 0.02:
            self.pos[1] = 0.02
            self.direction[1] = abs(self.direction[1]) * bounce_factor
        elif self.pos[1] > WORLD_SIZE - 0.02:
            self.pos[1] = WORLD_SIZE - 0.02
            self.direction[1] = -abs(self.direction[1]) * bounce_factor
        
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
    
    def generate_model_update(self):
        """Generate model update with potential poisoning"""
        if self.is_malicious:
            if self.attack_type == 'MODEL_POISONING':
                update = {
                    'drone_id': self.id,
                    'timestamp': time.time(),
                    'weights': np.random.randn(100) * self.attack_strength * 10,
                    'gradients': np.random.randn(100) * self.attack_strength * 5,
                    'is_poisoned': True,
                    'attack_strength': self.attack_strength
                }
            else:
                update = {
                    'drone_id': self.id,
                    'timestamp': time.time(),
                    'weights': np.random.randn(100) * (1 + self.attack_strength * 0.1),
                    'gradients': np.random.randn(100) * (1 + self.attack_strength * 0.05),
                    'is_poisoned': True,
                    'attack_type': self.attack_type
                }
        else:
            update = {
                'drone_id': self.id,
                'timestamp': time.time(),
                'weights': np.random.randn(100) * 0.1,
                'gradients': np.random.randn(100) * 0.05,
                'is_poisoned': False
            }
        
        # Store in history
        self.update_history.append(update)
        return update

# =======================
# STRONG DEFENSE ARCHITECTURE
# =======================
class DefenseArchitecture:
    """Implements strong defense mechanisms against attacks"""
    
    def __init__(self):
        self.trust_manager = ByzantineResistantTrustManager()
        self.detection_history = defaultdict(list)
        self.attack_patterns = self.initialize_attack_patterns()
        
    def initialize_attack_patterns(self):
        """Initialize known attack patterns"""
        return {
            'MODEL_POISONING': {
                'weight_deviation_threshold': 3.0,
                'gradient_magnitude_threshold': 2.5,
                'update_frequency_threshold': 0.8
            },
            'SYBIL': {
                'sei_similarity_threshold': 0.95,
                'spatial_proximity_threshold': 0.1,
                'time_synchronization_threshold': 0.05
            },
            'REPLAY': {
                'sei_consistency_threshold': 0.85,
                'timestamp_anomaly_threshold': 2.0,
                'signal_correlation_threshold': 0.7
            },
            'EVASIVE': {
                'sei_variability_threshold': 0.3,
                'movement_anomaly_threshold': 0.4,
                'communication_pattern_threshold': 0.6
            }
        }
    
    def detect_model_poisoning(self, update, drone):
        """Detect model poisoning attacks"""
        detection_score = 0.0
        reasons = []
        
        # Check weight deviations
        weights = update.get('weights', [])
        if len(weights) > 0:
            weight_std = np.std(weights)
            threshold = self.attack_patterns['MODEL_POISONING']['weight_deviation_threshold']
            if weight_std > threshold:
                detection_score += 0.4
                reasons.append(f"Weight deviation: {weight_std:.2f}")
        
        # Check gradient magnitude
        gradients = update.get('gradients', [])
        if len(gradients) > 0:
            grad_magnitude = np.linalg.norm(gradients)
            threshold = self.attack_patterns['MODEL_POISONING']['gradient_magnitude_threshold']
            if grad_magnitude > threshold:
                detection_score += 0.3
                reasons.append(f"Gradient magnitude: {grad_magnitude:.2f}")
        
        # Check update frequency
        if hasattr(drone, 'update_history') and drone.update_history:
            recent_updates = list(drone.update_history)
            if len(recent_updates) >= 2:
                intervals = np.diff([u['timestamp'] for u in recent_updates])
                avg_interval = np.mean(intervals) if len(intervals) > 0 else 10
                threshold = self.attack_patterns['MODEL_POISONING']['update_frequency_threshold']
                if avg_interval < threshold:
                    detection_score += 0.3
                    reasons.append(f"Suspicious update frequency: {avg_interval:.2f}")
        
        return detection_score > 0.5, detection_score, reasons
    
    def detect_sybil_attack(self, drones):
        """Detect Sybil attacks"""
        if len(drones) < 2:
            return False, 0.0, []
        
        detection_score = 0.0
        reasons = []
        
        # Group drones by SEI similarity
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                sei_sim = self.trust_manager.calculate_sei_similarity(
                    drones[i].sei_params, drones[j].sei_params
                )
                
                if sei_sim > self.attack_patterns['SYBIL']['sei_similarity_threshold']:
                    # Check spatial proximity
                    distance = euclidean(drones[i].pos, drones[j].pos)
                    if distance < self.attack_patterns['SYBIL']['spatial_proximity_threshold']:
                        detection_score += 0.5
                        reasons.append(f"Sybil cluster: {drones[i].id} & {drones[j].id}")
        
        return detection_score > 0.0, detection_score, reasons
    
    def detect_replay_attack(self, drone, recent_signals):
        """Detect replay attacks"""
        if len(recent_signals) < 3:
            return False, 0.0, []
        
        detection_score = 0.0
        reasons = []
        
        # Check SEI consistency
        sei_values = [s.get('sei_params', {}) for s in recent_signals]
        cfo_values = [s.get('cfo_hz', 0) for s in sei_values if 'cfo_hz' in s]
        
        if len(cfo_values) >= 2:
            cfo_std = np.std(cfo_values)
            if cfo_std < 10:
                detection_score += 0.4
                reasons.append(f"Suspicious SEI consistency: {cfo_std:.2f}")
        
        return detection_score > 0.5, detection_score, reasons
    
    def comprehensive_security_check(self, drone, update=None):
        """Perform comprehensive security check"""
        security_report = {
            'drone_id': drone.id,
            'trust_score': self.trust_manager.trust_scores.get(drone.id, 0.5),
            'verifications': [],
            'detections': [],
            'recommendation': 'ALLOW'
        }
        
        # SEI verification
        sei_verified, sei_similarity = self.trust_manager.verify_drone_identity(
            drone.id, drone.sei_params, drone.id
        )
        security_report['verifications'].append({
            'type': 'SEI',
            'passed': sei_verified,
            'similarity': sei_similarity
        })
        
        # Attack detection
        if drone.is_malicious:
            if update and drone.attack_type == 'MODEL_POISONING':
                detected, score, reasons = self.detect_model_poisoning(update, drone)
                if detected:
                    security_report['detections'].append({
                        'type': 'MODEL_POISONING',
                        'confidence': score,
                        'reasons': reasons
                    })
            
            if drone.attack_type == 'SYBIL':
                detected, score, reasons = self.detect_sybil_attack(st.session_state.drones)
                if detected:
                    security_report['detections'].append({
                        'type': 'SYBIL',
                        'confidence': score,
                        'reasons': reasons
                    })
        
        # Determine recommendation
        if not sei_verified or len(security_report['detections']) > 0:
            security_report['recommendation'] = 'ISOLATE'
        elif security_report['trust_score'] < MIN_TRUST_SCORE:
            security_report['recommendation'] = 'MONITOR'
        
        return security_report

# =======================
# SIMULATION MANAGEMENT
# =======================
def initialize_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.simulation_running = True
        st.session_state.last_update_time = time.time()
        st.session_state.update_count = 0
        
        # Load dataset
        st.session_state.dataset_samples = load_dataset_samples()
        
        # Create initial drones from dataset
        st.session_state.drones = create_initial_drones()
        st.session_state.selected_drone = None
        st.session_state.drone_counter = INITIAL_DRONES
        
        # Initialize defense architecture
        st.session_state.defense = DefenseArchitecture()
        
        # Communication tracking
        st.session_state.communications = []
        st.session_state.malicious_drones = set()
        st.session_state.isolated_drones = set()
        
        # Update tracking
        st.session_state.update_vectors = []
        st.session_state.security_reports = []
        
        # Logs
        st.session_state.logs = deque(maxlen=100)
        
        # Add initial log
        add_log(f"Initialized with {len(st.session_state.drones)} drones from dataset")
        add_log(f"Dataset contains: {list(st.session_state.dataset_samples.keys())}")

def create_initial_drones():
    """Create initial drones from dataset"""
    drones = []
    samples_by_type = st.session_state.dataset_samples
    
    # Priority drone types to include
    priority_types = ['DJI', 'FUTABA', 'ENEMY', 'UNKNOWN']
    
    drone_count = 0
    for drone_type in priority_types:
        # Find matching types
        matching_types = [t for t in samples_by_type.keys() if drone_type in t]
        for match_type in matching_types:
            if samples_by_type[match_type]:
                sample = random.choice(samples_by_type[match_type])
                
                # Determine if trusted or malicious
                is_trusted = 'ENEMY' not in match_type
                is_malicious = 'ENEMY' in match_type
                
                drone = AdvancedDrone(
                    drone_id=f"D{drone_count+1}",
                    drone_type=match_type,
                    sample_data=sample,
                    is_trusted=is_trusted,
                    is_malicious=is_malicious
                )
                drones.append(drone)
                drone_count += 1
                
                if drone_count >= INITIAL_DRONES:
                    return drones
    
    # If we need more drones, add random types
    while drone_count < INITIAL_DRONES:
        available_types = list(samples_by_type.keys())
        if not available_types:
            break
            
        drone_type = random.choice(available_types)
        if samples_by_type[drone_type]:
            sample = random.choice(samples_by_type[drone_type])
            
            is_trusted = 'ENEMY' not in drone_type
            is_malicious = 'ENEMY' in drone_type
            
            drone = AdvancedDrone(
                drone_id=f"D{drone_count+1}",
                drone_type=drone_type,
                sample_data=sample,
                is_trusted=is_trusted,
                is_malicious=is_malicious
            )
            drones.append(drone)
            drone_count += 1
    
    return drones

def add_log(message, level="INFO"):
    """Add a log message"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Determine emoji and style
    if "ATTACK" in message or "MALICIOUS" in message:
        emoji = "🚨"
    elif "DETECTED" in message or "BLOCKED" in message or "SECURITY" in message:
        emoji = "🛡️"
    elif "WARNING" in message:
        emoji = "⚠️"
    elif "TRUSTED" in message or "VERIFIED" in message:
        emoji = "✅"
    elif "UPDATE" in message:
        emoji = "🔄"
    elif "ADDED" in message:
        emoji = "🆕"
    else:
        emoji = "ℹ️"
    
    log_entry = f"[{timestamp}] {emoji} {message}"
    st.session_state.logs.append(log_entry)
    
    # Write to file
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except:
        pass

def update_simulation():
    """Update simulation state"""
    if not st.session_state.simulation_running:
        return
    
    current_time = time.time()
    if current_time - st.session_state.last_update_time < UPDATE_INTERVAL:
        return
    
    st.session_state.last_update_time = current_time
    st.session_state.update_count += 1
    
    # Move all drones
    for drone in st.session_state.drones:
        drone.move()
    
    # Clear previous communications
    st.session_state.communications = []
    
    # Update communications and security checks
    for i in range(len(st.session_state.drones)):
        for j in range(i + 1, len(st.session_state.drones)):
            drone_a = st.session_state.drones[i]
            drone_b = st.session_state.drones[j]
            
            # Skip if isolated
            if drone_a.isolated or drone_b.isolated:
                continue
            
            # Calculate distance
            distance = euclidean(drone_a.pos, drone_b.pos)
            
            # If in communication range
            if distance < COMM_RANGE:
                # Record communication
                st.session_state.communications.append((drone_a, drone_b, distance))
    
    # Handle model updates and attack detection
    if st.session_state.update_count % 5 == 0:
        for drone in st.session_state.drones:
            if not drone.isolated and time.time() - drone.last_update_time > drone.update_interval:
                # Generate update
                update = drone.generate_model_update()
                drone.last_update_time = time.time()
                
                # Security check
                security_report = st.session_state.defense.comprehensive_security_check(drone, update)
                st.session_state.security_reports.append(security_report)
                
                # Log based on security check - FIXED: Use drone.label instead of drone.label
                if security_report['recommendation'] == 'ISOLATE':
                    drone.isolated = True
                    detection_text = ""
                    if security_report['detections']:
                        detection_text = str([d['type'] for d in security_report['detections']])
                    add_log(f"ATTACK DETECTED: {drone.label} - {detection_text}", "ATTACK")
                elif any(d['type'] == 'MODEL_POISONING' for d in security_report.get('detections', [])):
                    add_log(f"POISONING DETECTED: {drone.label} - Update rejected", "ATTACK")
                else:
                    add_log(f"UPDATE: {drone.label} → GBS | Trust: {security_report['trust_score']:.2f}")

# =======================
# VISUALIZATION FUNCTIONS
# =======================
def create_swarm_visualization():
    """Create visualization of the swarm"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Swarm visualization
    ax1.set_facecolor('#0f0f23')
    ax1.set_xlim(-0.05, WORLD_SIZE + 0.05)
    ax1.set_ylim(-0.05, WORLD_SIZE + 0.05)
    ax1.set_aspect('equal')
    ax1.set_title('UAV Swarm - Q-SEI-FED Defense System', color='white', fontsize=14, pad=15)
    ax1.set_xlabel('X Position', color='white')
    ax1.set_ylabel('Y Position', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    
    # Plot GBS at center
    gbs_pos = np.array([WORLD_SIZE/2, WORLD_SIZE/2])
    ax1.scatter(gbs_pos[0], gbs_pos[1], s=400, c=COLORS['gbs'], marker='^',
               edgecolors='white', linewidths=2, alpha=0.9, zorder=30)
    ax1.text(gbs_pos[0], gbs_pos[1] + 0.05, 'GBS', fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.8),
            color='white', fontweight='bold', zorder=35)
    
    # Plot communication ranges
    for drone in st.session_state.drones:
        if not drone.isolated:
            circle = plt.Circle(drone.pos, COMM_RANGE, 
                              color='blue', alpha=0.03, fill=True)
            ax1.add_patch(circle)
    
    # Plot communication lines
    for comm in st.session_state.communications:
        drone_a, drone_b, distance = comm
        color = 'cyan' if drone_a.trusted and drone_b.trusted else 'red'
        style = '-' if drone_a.trusted and drone_b.trusted else '--'
        alpha = 0.6 if distance < COMM_RANGE else 0.3
        
        ax1.plot([drone_a.pos[0], drone_b.pos[0]], 
                [drone_a.pos[1], drone_b.pos[1]], 
                color=color, linestyle=style, alpha=alpha, linewidth=1.5)
    
    # Plot drones
    for drone in st.session_state.drones:
        if drone.isolated:
            color = COLORS['isolated']
            marker = 'X'
            size = 350
            zorder = 25
        elif drone.is_malicious:
            color = COLORS['malicious']
            marker = 's'
            size = 300
            zorder = 20
        elif not drone.trusted:
            color = COLORS['enemy']
            marker = 'd'
            size = 280
            zorder = 15
        else:
            color = COLORS['trusted']
            marker = 'o'
            size = 250
            zorder = 10
        
        ax1.scatter(drone.pos[0], drone.pos[1], s=size, c=color, marker=marker,
                   edgecolors='black', linewidths=1.5, alpha=0.9, zorder=zorder)
        
        # Add label with trust score
        trust_score = st.session_state.defense.trust_manager.trust_scores.get(drone.id, 0.5)
        ax1.text(drone.pos[0], drone.pos[1] - 0.045, f"{drone.drone_type}\n{trust_score:.2f}",
                fontsize=7, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                color='white', zorder=30)
    
    # Right: Security status visualization
    ax2.set_facecolor('#0f0f23')
    
    # Trust score distribution
    if st.session_state.drones:
        trust_scores = []
        drone_types = []
        colors = []
        
        for drone in st.session_state.drones:
            score = st.session_state.defense.trust_manager.trust_scores.get(drone.id, 0.5)
            trust_scores.append(score)
            drone_types.append(drone.drone_type[:10])
            
            if drone.isolated:
                colors.append(COLORS['isolated'])
            elif drone.is_malicious:
                colors.append(COLORS['malicious'])
            elif not drone.trusted:
                colors.append(COLORS['enemy'])
            else:
                colors.append(COLORS['trusted'])
        
        y_pos = np.arange(len(trust_scores))
        bars = ax2.barh(y_pos, trust_scores, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(drone_types, fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Trust Score', color='white')
        ax2.set_title('Drone Trust Scores', color='white', fontsize=12, pad=10)
        ax2.axvline(x=MIN_TRUST_SCORE, color='red', linestyle='--', alpha=0.5, label='Min Trust')
        ax2.axvline(x=SEI_MATCH_THRESHOLD, color='yellow', linestyle='--', alpha=0.5, label='SEI Threshold')
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
        ax2.grid(True, alpha=0.2, axis='x')
    
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#1a1a2e')
    
    plt.tight_layout()
    return fig

def plot_sei_analysis(drone):
    """Plot SEI analysis for a drone"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # 1. SEI Parameters Radar
    sei_params = drone.sei_params
    param_names = ['CFO (Hz)', 'IQ Gain', 'IQ Phase', 'PA Alpha', 'Phase Noise', 'Filter Fc']
    param_values = [
        abs(sei_params.get('cfo_hz', 0)) / 1000,
        sei_params.get('iq_gain', 1),
        abs(sei_params.get('iq_phase', 0)) / 10,
        sei_params.get('pa_alpha', 2.5) / 3,
        sei_params.get('phase_noise', 5) / 15,
        sei_params.get('filter_fc', 1)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
    param_values += param_values[:1]
    angles += angles[:1]
    
    ax_radar = plt.subplot(2, 3, 1, polar=True)
    ax_radar.plot(angles, param_values, 'o-', linewidth=2)
    ax_radar.fill(angles, param_values, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(param_names)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('SEI Parameters Radar', fontsize=10)
    
    # 2. IQ Signal
    iq_data = drone.iq_data[:2000]
    axes[1].plot(np.real(iq_data), 'b-', label='I', alpha=0.7)
    axes[1].plot(np.imag(iq_data), 'r-', label='Q', alpha=0.7)
    axes[1].set_title('IQ Signal (First 2000 samples)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Constellation
    axes[2].scatter(np.real(iq_data), np.imag(iq_data), s=10, alpha=0.6)
    axes[2].set_title('Constellation Diagram')
    axes[2].set_xlabel('I')
    axes[2].set_ylabel('Q')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    # 4. Spectrogram
    if hasattr(drone, 'spectrogram_data') and drone.spectrogram_data is not None:
        spec = drone.spectrogram_data
        if spec.ndim == 2:
            im = axes[3].imshow(spec[:64, :100], aspect='auto', origin='lower', cmap='hot')
            axes[3].set_title('Spectrogram')
            axes[3].set_xlabel('Time')
            axes[3].set_ylabel('Frequency')
            plt.colorbar(im, ax=axes[3])
    
    # 5. Trust History
    if drone.id in st.session_state.defense.trust_manager.trust_history:
        history = list(st.session_state.defense.trust_manager.trust_history[drone.id])
        if history:
            timestamps = [h['timestamp'] for h in history]
            scores = [h['score'] for h in history]
            axes[4].plot(timestamps[-20:], scores[-20:], 'g-', marker='o', markersize=3)
            axes[4].axhline(y=MIN_TRUST_SCORE, color='r', linestyle='--', alpha=0.5)
            axes[4].axhline(y=SEI_MATCH_THRESHOLD, color='y', linestyle='--', alpha=0.5)
            axes[4].set_title('Trust Score History')
            axes[4].set_xlabel('Time')
            axes[4].set_ylabel('Trust Score')
            axes[4].grid(True, alpha=0.3)
            axes[4].set_ylim(0, 1)
    
    # 6. Security Status
    axes[5].axis('off')
    security_text = f"Drone ID: {drone.id}\n"
    security_text += f"Type: {drone.drone_type}\n"
    security_text += f"Trusted: {drone.trusted}\n"
    security_text += f"Malicious: {drone.is_malicious}\n"
    security_text += f"Isolated: {drone.isolated}\n"
    security_text += f"SEI Fingerprint: {drone.sei_fingerprint[:8]}...\n"
    
    if drone.is_malicious:
        security_text += f"\nAttack Type: {drone.attack_type}\n"
        security_text += f"Attack Strength: {drone.attack_strength:.2f}"
    
    axes[5].text(0.1, 0.5, security_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'SEI Analysis: {drone.label}', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

def plot_attack_detection():
    """Plot attack detection statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Get recent security reports
    recent_reports = st.session_state.security_reports[-20:] if st.session_state.security_reports else []
    
    if recent_reports:
        # 1. Attack types detected
        attack_types = []
        for report in recent_reports:
            for detection in report.get('detections', []):
                attack_types.append(detection['type'])
        
        if attack_types:
            attack_counts = Counter(attack_types)
            axes[0].bar(attack_counts.keys(), attack_counts.values(), color=['red', 'orange', 'purple', 'brown'])
            axes[0].set_title('Attack Types Detected')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
        else:
            axes[0].text(0.5, 0.5, 'No attacks detected', ha='center', va='center')
            axes[0].set_title('Attack Types Detected')
        
        # 2. Trust score distribution
        trust_scores = [r['trust_score'] for r in recent_reports]
        axes[1].hist(trust_scores, bins=10, color='green', alpha=0.7)
        axes[1].axvline(x=MIN_TRUST_SCORE, color='r', linestyle='--', label='Min Trust')
        axes[1].set_title('Trust Score Distribution')
        axes[1].set_xlabel('Trust Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # 3. Detection confidence
        confidences = []
        for report in recent_reports:
            for detection in report.get('detections', []):
                confidences.append(detection['confidence'])
        
        if confidences:
            axes[2].hist(confidences, bins=10, color='blue', alpha=0.7)
            axes[2].set_title('Detection Confidence')
            axes[2].set_xlabel('Confidence')
            axes[2].set_ylabel('Frequency')
        else:
            axes[2].text(0.5, 0.5, 'No detections', ha='center', va='center')
            axes[2].set_title('Detection Confidence')
    
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No security data yet', ha='center', va='center')
            ax.set_title(['Attack Types', 'Trust Distribution', 'Detection Confidence'][list(axes).index(ax)])
    
    plt.suptitle('Security Dashboard', fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

# =======================
# ATTACK SIMULATION FUNCTIONS
# =======================
def simulate_advanced_poisoning_attack():
    """Simulate advanced model poisoning attack"""
    trusted_drones = [d for d in st.session_state.drones 
                     if d.trusted and not d.is_malicious and not d.isolated]
    
    if not trusted_drones:
        add_log("No trusted drones available for poisoning", "WARNING")
        return
    
    target = random.choice(trusted_drones)
    target.poisoned = True
    target.is_malicious = True
    target.attack_type = 'MODEL_POISONING'
    target.attack_strength = random.uniform(0.7, 1.0)
    
    # Slightly modify SEI to evade detection
    target.sei_params['cfo_hz'] *= random.uniform(0.95, 1.05)
    target.sei_params['iq_gain'] *= random.uniform(0.98, 1.02)
    
    add_log(f"ADVANCED POISONING: {target.label} compromised with strength {target.attack_strength:.2f}", "ATTACK")
    
    # Generate poisoned update
    update = target.generate_model_update()
    
    # Try to send update
    security_report = st.session_state.defense.comprehensive_security_check(target, update)
    
    if security_report['recommendation'] == 'ISOLATE':
        add_log(f"POISONING DETECTED: {target.label} isolated by defense system", "ATTACK")
        target.isolated = True
    else:
        add_log(f"POISONING EVADED: {target.label} update accepted (should be detected)", "WARNING")

def simulate_sybil_swarm_attack():
    """Simulate coordinated Sybil swarm attack"""
    if len(st.session_state.drones) >= MAX_DRONES - 3:
        add_log("Not enough room for Sybil swarm", "WARNING")
        return
    
    # Create multiple drones with similar SEI
    base_sample = None
    for drone in st.session_state.drones:
        if drone.trusted and not drone.is_malicious:
            base_sample = drone.sample_data
            break
    
    if not base_sample:
        add_log("No base drone for Sybil attack", "WARNING")
        return
    
    num_sybil = random.randint(2, 3)
    created = 0
    
    for i in range(num_sybil):
        if len(st.session_state.drones) >= MAX_DRONES:
            break
        
        st.session_state.drone_counter += 1
        
        # Clone SEI parameters with slight variations
        cloned_sei = base_sample['sei_params'].copy()
        cloned_sei['cfo_hz'] *= random.uniform(0.99, 1.01)
        cloned_sei['iq_gain'] *= random.uniform(0.995, 1.005)
        
        sybil_sample = base_sample.copy()
        sybil_sample['sei_params'] = cloned_sei
        
        sybil_drone = AdvancedDrone(
            drone_id=f"S{st.session_state.drone_counter}",
            drone_type=f"Sybil_{base_sample['drone_type']}",
            sample_data=sybil_sample,
            is_trusted=False,
            is_malicious=True
        )
        sybil_drone.attack_type = 'SYBIL'
        sybil_drone.attack_strength = random.uniform(0.8, 1.0)
        
        # Position near each other
        base_pos = random.choice(st.session_state.drones).pos if st.session_state.drones else np.array([0.5, 0.5])
        sybil_drone.pos = base_pos + np.random.randn(2) * 0.05
        
        st.session_state.drones.append(sybil_drone)
        created += 1
    
    if created > 0:
        add_log(f"SYBIL SWARM: {created} Sybil drones created", "ATTACK")

def simulate_replay_evasion_attack():
    """Simulate replay attack with evasion techniques"""
    trusted_drones = [d for d in st.session_state.drones 
                     if d.trusted and not d.is_malicious and 'UNKNOWN' not in d.drone_type]
    
    if not trusted_drones:
        add_log("No suitable drone for replay attack", "WARNING")
        return
    
    original = random.choice(trusted_drones)
    
    # Create replay drone
    st.session_state.drone_counter += 1
    
    # Copy but modify SEI slightly
    replay_sei = original.sei_params.copy()
    replay_sei['cfo_hz'] = original.sei_params.get('cfo_hz', 0) + random.uniform(-50, 50)
    replay_sei['phase_noise'] = original.sei_params.get('phase_noise', 5) * random.uniform(0.9, 1.1)
    
    replay_sample = original.sample_data.copy()
    replay_sample['sei_params'] = replay_sei
    replay_sample['drone_type'] = f"Replay_{original.drone_type}"
    
    replay_drone = AdvancedDrone(
        drone_id=f"R{st.session_state.drone_counter}",
        drone_type=replay_sample['drone_type'],
        sample_data=replay_sample,
        is_trusted=False,
        is_malicious=True
    )
    replay_drone.attack_type = 'REPLAY'
    replay_drone.attack_strength = random.uniform(0.6, 0.9)
    
    # Position near original
    replay_drone.pos = original.pos + np.random.randn(2) * 0.03
    
    if len(st.session_state.drones) < MAX_DRONES:
        st.session_state.drones.append(replay_drone)
        add_log(f"REPLAY ATTACK: Clone of {original.label} created", "ATTACK")

# =======================
# MAIN APPLICATION
# =======================
def main():
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 2px solid #4ECDC4;
    }
    
    .attack-button {
        background: linear-gradient(45deg, #FF416C, #FF4B2B) !important;
        color: white !important;
        border: 2px solid #FF0000 !important;
        font-weight: bold !important;
    }
    
    .defense-button {
        background: linear-gradient(45deg, #11998e, #38ef7d) !important;
        color: white !important;
        border: 2px solid #00FF00 !important;
        font-weight: bold !important;
    }
    
    .control-button {
        background: linear-gradient(45deg, #4A00E0, #8E2DE2) !important;
        color: white !important;
        border: 2px solid #9400D3 !important;
        font-weight: bold !important;
    }
    
    .metric-card {
        background: rgba(30, 30, 46, 0.9);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 5px solid #4ECDC4;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .drone-card {
        background: rgba(40, 40, 60, 0.8);
        border-radius: 10px;
        padding: 12px;
        margin: 6px 0;
        border-left: 4px solid #48bb78;
        transition: all 0.3s ease;
        color: white;
    }
    
    .drone-card:hover {
        background: rgba(50, 50, 80, 0.9);
        transform: translateX(5px);
    }
    
    .malicious-card {
        border-left: 4px solid #FF6B6B !important;
        background: rgba(255, 107, 107, 0.15) !important;
    }
    
    .isolated-card {
        border-left: 4px solid #FFA500 !important;
        background: rgba(255, 165, 0, 0.15) !important;
    }
    
    .log-container {
        background: rgba(20, 20, 35, 0.9);
        border-radius: 10px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #2d3748;
    }
    
    .log-entry {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 0.85em;
        font-family: 'Courier New', monospace;
        background: rgba(30, 30, 50, 0.6);
        color: #e0e0e0;
        border-left: 3px solid #0096FF;
    }
    
    .log-attack {
        border-left: 3px solid #FF00FF !important;
        background: rgba(255, 0, 255, 0.1) !important;
        font-weight: bold;
    }
    
    .log-defense {
        border-left: 3px solid #00FF00 !important;
        background: rgba(0, 255, 0, 0.1) !important;
    }
    
    .log-warning {
        border-left: 3px solid #FFA500 !important;
        background: rgba(255, 165, 0, 0.1) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1a2e;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        border-radius: 8px;
        padding: 10px 20px;
        color: #a0aec0;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a5568 !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Update simulation
    update_simulation()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🛡️ Q-SEI-FED: Quantum-Resistant UAV Swarm Security</h1>
        <p style="color: #a0aec0; margin: 5px 0 0 0;">
        <strong>Specific Emitter Identification + Byzantine-Resistant Federated Learning</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_drones = len(st.session_state.drones)
        st.metric("Total Drones", total_drones, f"Max: {MAX_DRONES}")
    
    with col2:
        trusted = sum(1 for d in st.session_state.drones if d.trusted and not d.isolated)
        st.metric("Trusted", trusted)
    
    with col3:
        malicious = sum(1 for d in st.session_state.drones if d.is_malicious)
        st.metric("Malicious", malicious)
    
    with col4:
        isolated = sum(1 for d in st.session_state.drones if d.isolated)
        st.metric("Isolated", isolated)
    
    with col5:
        if st.session_state.security_reports:
            recent = st.session_state.security_reports[-5:] if len(st.session_state.security_reports) >= 5 else st.session_state.security_reports
            detected = sum(1 for r in recent if r.get('detections'))
            st.metric("Recent Detections", detected, f"of {len(recent)}")
        else:
            st.metric("Detections", 0)
    
    # Main layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Swarm visualization
        st.markdown("### 🛰️ Swarm Visualization & Security Status")
        swarm_fig = create_swarm_visualization()
        st.pyplot(swarm_fig, use_container_width=True)
        plt.close(swarm_fig)
        
        # Attack simulation panel
        st.markdown("### ⚔️ Advanced Attack Simulation")
        
        attack_col1, attack_col2, attack_col3 = st.columns(3)
        
        with attack_col1:
            if st.button("🧪 Advanced Poisoning", key="adv_poisoning", 
                        use_container_width=True, type="secondary"):
                simulate_advanced_poisoning_attack()
                time.sleep(0.1)
                st.rerun()
        
        with attack_col2:
            if st.button("👥 Sybil Swarm", key="sybil_swarm",
                        use_container_width=True, type="secondary"):
                simulate_sybil_swarm_attack()
                time.sleep(0.1)
                st.rerun()
        
        with attack_col3:
            if st.button("🔄 Replay Evasion", key="replay_evasion",
                        use_container_width=True, type="secondary"):
                simulate_replay_evasion_attack()
                time.sleep(0.1)
                st.rerun()
        
        # Drone management
        st.markdown("### 🛠️ Drone Management")
        
        mgmt_col1, mgmt_col2, mgmt_col3, mgmt_col4 = st.columns(4)
        
        with mgmt_col1:
            if len(st.session_state.drones) < MAX_DRONES:
                if st.button("➕ Add Trusted", key="add_trusted",
                            use_container_width=True):
                    trusted_types = [t for t in st.session_state.dataset_samples.keys() 
                                   if 'ENEMY' not in t and 'UNKNOWN' not in t]
                    if trusted_types:
                        drone_type = random.choice(trusted_types)
                        if st.session_state.dataset_samples[drone_type]:
                            sample = random.choice(st.session_state.dataset_samples[drone_type])
                            st.session_state.drone_counter += 1
                            new_drone = AdvancedDrone(
                                drone_id=f"D{st.session_state.drone_counter}",
                                drone_type=drone_type,
                                sample_data=sample,
                                is_trusted=True,
                                is_malicious=False
                            )
                            st.session_state.drones.append(new_drone)
                            add_log(f"ADDED: Trusted {drone_type} drone")
                    time.sleep(0.1)
                    st.rerun()
        
        with mgmt_col2:
            if len(st.session_state.drones) < MAX_DRONES:
                if st.button("➖ Add Malicious", key="add_malicious",
                            use_container_width=True):
                    enemy_types = [t for t in st.session_state.dataset_samples.keys() 
                                 if 'ENEMY' in t]
                    if not enemy_types:
                        enemy_types = list(st.session_state.dataset_samples.keys())
                    
                    if enemy_types:
                        drone_type = random.choice(enemy_types)
                        if st.session_state.dataset_samples[drone_type]:
                            sample = random.choice(st.session_state.dataset_samples[drone_type])
                            st.session_state.drone_counter += 1
                            new_drone = AdvancedDrone(
                                drone_id=f"M{st.session_state.drone_counter}",
                                drone_type=drone_type,
                                sample_data=sample,
                                is_trusted=False,
                                is_malicious=True
                            )
                            st.session_state.drones.append(new_drone)
                            add_log(f"ADDED: Malicious {drone_type} drone", "WARNING")
                    time.sleep(0.1)
                    st.rerun()
        
        with mgmt_col3:
            if st.session_state.selected_drone:
                selected = next((d for d in st.session_state.drones 
                               if d.id == st.session_state.selected_drone), None)
                if selected and selected.trusted:
                    if st.button("🤖 Compromise", key="compromise",
                                use_container_width=True):
                        selected.trusted = False
                        selected.is_malicious = True
                        selected.attack_type = random.choice(['MODEL_POISONING', 'EVASIVE'])
                        selected.attack_strength = random.uniform(0.7, 1.0)
                        add_log(f"COMPROMISED: {selected.label} turned malicious", "ATTACK")
                        time.sleep(0.1)
                        st.rerun()
        
        with mgmt_col4:
            btn_text = "⏸️ Pause" if st.session_state.simulation_running else "▶️ Resume"
            if st.button(btn_text, key="pause_resume", use_container_width=True):
                st.session_state.simulation_running = not st.session_state.simulation_running
                action = "paused" if not st.session_state.simulation_running else "resumed"
                add_log(f"Simulation {action}")
                time.sleep(0.1)
                st.rerun()
        
        # Defense actions
        st.markdown("### 🛡️ Defense Actions")
        
        defense_col1, defense_col2, defense_col3 = st.columns(3)
        
        with defense_col1:
            if st.button("🔍 Security Scan", key="security_scan",
                        use_container_width=True, type="primary"):
                for drone in st.session_state.drones:
                    report = st.session_state.defense.comprehensive_security_check(drone)
                    if report['recommendation'] == 'ISOLATE' and not drone.isolated:
                        drone.isolated = True
                        add_log(f"SECURITY SCAN: {drone.label} isolated", "ATTACK")
                add_log("Security scan completed")
                time.sleep(0.1)
                st.rerun()
        
        with defense_col2:
            if st.button("🔄 Send Updates", key="send_updates",
                        use_container_width=True, type="primary"):
                updates_sent = 0
                blocked = 0
                
                for drone in st.session_state.drones:
                    if not drone.isolated:
                        update = drone.generate_model_update()
                        report = st.session_state.defense.comprehensive_security_check(drone, update)
                        
                        if report['recommendation'] == 'ISOLATE':
                            blocked += 1
                            add_log(f"UPDATE BLOCKED: {drone.label} - {report.get('detections', [])}", "ATTACK")
                        else:
                            updates_sent += 1
                            st.session_state.update_vectors.append(update)
                
                add_log(f"FEDERATED LEARNING: {updates_sent} updates sent, {blocked} blocked")
                time.sleep(0.1)
                st.rerun()
        
        with defense_col3:
            if st.button("🧹 Clear Logs", key="clear_logs",
                        use_container_width=True):
                st.session_state.logs.clear()
                add_log("Logs cleared")
                time.sleep(0.1)
                st.rerun()
        
        # Security dashboard
        st.markdown("### 📊 Security Dashboard")
        attack_fig = plot_attack_detection()
        st.pyplot(attack_fig, use_container_width=True)
        plt.close(attack_fig)
    
    with col_right:
        # Drone list
        st.markdown("### 🛸 Active Drones")
        
        for drone in st.session_state.drones:
            trust_level = st.session_state.defense.trust_manager.get_trust_level(drone.id)
            
            # Determine card style
            if drone.isolated:
                status = "🔴 ISOLATED"
                card_class = "drone-card isolated-card"
            elif drone.is_malicious:
                status = "⚠️ MALICIOUS"
                card_class = "drone-card malicious-card"
            elif trust_level == "SUSPICIOUS":
                status = "🔍 SUSPICIOUS"
                card_class = "drone-card"
            else:
                status = "✅ TRUSTED"
                card_class = "drone-card"
            
            # Create expandable card
            with st.expander(f"{drone.drone_type} - {status}", expanded=False):
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    trust_score = st.session_state.defense.trust_manager.trust_scores.get(drone.id, 0.5)
                    st.write(f"**ID:** {drone.id}")
                    st.write(f"**Type:** {drone.drone_type}")
                    st.write(f"**Trust Score:** {trust_score:.3f}")
                    st.write(f"**Position:** ({drone.pos[0]:.3f}, {drone.pos[1]:.3f})")
                    
                    if drone.is_malicious:
                        st.write(f"**Attack:** {drone.attack_type}")
                        st.write(f"**Strength:** {drone.attack_strength:.2f}")
                
                with col_b:
                    if st.button("Select", key=f"select_{drone.id}"):
                        st.session_state.selected_drone = drone.id
                        time.sleep(0.1)
                        st.rerun()
                
                # Quick actions for selected drone
                if drone.id == st.session_state.selected_drone:
                    st.success(f"✓ {drone.label} selected")
        
        # Log display
        st.markdown("### 📋 Security Log")
        
        log_container = st.container(height=350)
        
        with log_container:
            # Display logs
            for log in reversed(list(st.session_state.logs)[-15:]):
                if "ATTACK" in log or "MALICIOUS" in log:
                    log_class = "log-attack"
                elif "DETECTED" in log or "BLOCKED" in log or "SECURITY" in log:
                    log_class = "log-defense"
                elif "WARNING" in log:
                    log_class = "log-warning"
                else:
                    log_class = "log-entry"
                
                st.markdown(f'<div class="{log_class}">{log}</div>', unsafe_allow_html=True)
        
        # Defense metrics
        st.markdown("### 📈 Defense Metrics")
        
        total_updates = len(st.session_state.update_vectors)
        recent_reports = st.session_state.security_reports[-10:] if st.session_state.security_reports else []
        
        if recent_reports:
            attacks_detected = sum(1 for r in recent_reports if r.get('detections'))
            avg_trust = np.mean([r['trust_score'] for r in recent_reports]) if recent_reports else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Recent Performance:</strong></p>
                <p>• Updates Processed: <strong>{total_updates}</strong></p>
                <p>• Attacks Detected: <strong>{attacks_detected}</strong></p>
                <p>• Avg Trust Score: <strong>{avg_trust:.3f}</strong></p>
                <p>• SEI Threshold: <strong>{SEI_MATCH_THRESHOLD}</strong></p>
                <p>• Min Consensus: <strong>{MIN_CONSENSUS_RATIO*100:.0f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <p><strong>Waiting for data...</strong></p>
                <p>Simulation metrics will appear here</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis for selected drone
    if st.session_state.selected_drone:
        selected = next((d for d in st.session_state.drones 
                       if d.id == st.session_state.selected_drone), None)
        
        if selected:
            st.markdown("---")
            st.markdown(f"## 🔬 Detailed Analysis: {selected.drone_type}")
            
            # Create tabs
            tab1, tab2 = st.tabs(["📡 SEI Analysis", "🎯 Security Report"])
            
            with tab1:
                # SEI analysis
                sei_fig = plot_sei_analysis(selected)
                st.pyplot(sei_fig, use_container_width=True)
                plt.close(sei_fig)
                
                # SEI parameters table
                st.markdown("#### SEI Parameters")
                sei_df = pd.DataFrame.from_dict(selected.sei_params, orient='index', columns=['Value'])
                st.dataframe(sei_df, use_container_width=True)
            
            with tab2:
                # Security report
                report = st.session_state.defense.comprehensive_security_check(selected)
                
                col_rep1, col_rep2 = st.columns(2)
                
                with col_rep1:
                    st.metric("Trust Score", f"{report['trust_score']:.3f}")
                    st.metric("Recommendation", report['recommendation'])
                    
                    # Trust level gauge
                    fig_gauge, ax_gauge = plt.subplots(figsize=(6, 2))
                    ax_gauge.barh([0], [1], color='lightgray', alpha=0.3, height=0.3)
                    ax_gauge.barh([0], [report['trust_score']], color='green' if report['trust_score'] > 0.6 else 'red', 
                                 height=0.3)
                    ax_gauge.axvline(x=MIN_TRUST_SCORE, color='red', linestyle='--', alpha=0.7)
                    ax_gauge.axvline(x=SEI_MATCH_THRESHOLD, color='yellow', linestyle='--', alpha=0.7)
                    ax_gauge.set_xlim(0, 1)
                    ax_gauge.set_yticks([])
                    ax_gauge.set_title('Trust Score Gauge')
                    ax_gauge.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig_gauge, use_container_width=True)
                    plt.close(fig_gauge)
                
                with col_rep2:
                    st.write("**Verifications:**")
                    for verif in report.get('verifications', []):
                        status = "✅" if verif['passed'] else "❌"
                        st.write(f"{status} {verif['type']}: {verif.get('similarity', verif.get('score', 0)):.3f}")
                    
                    st.write("**Detections:**")
                    if report.get('detections'):
                        for det in report['detections']:
                            st.write(f"🚨 {det['type']} (Confidence: {det['confidence']:.2f})")
                    else:
                        st.write("No attacks detected")
    
    # Auto-refresh for smooth simulation
    if st.session_state.simulation_running:
        time.sleep(UPDATE_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()