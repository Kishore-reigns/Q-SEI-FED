import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from typing import Tuple, List, Dict, Optional
warnings.filterwarnings('ignore')

# Set matplotlib to use Agg backend
matplotlib.use("Agg")

#setting up full width
st.set_page_config(
    page_title="Q-SEI-FED UAV Swarm",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =======================
# LSNet MODEL DEFINITION
# =======================
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

# =======================
# CONFIGURATION
# =======================
DATASET_ROOT = r"F:\K DRIVE\MIT_Learnings\Sem8\dataset\code\sei_dataset4"
LOG_FILE_PATH = "uav_simulation_logs.txt"
MAX_DRONES = 20
INITIAL_DRONES = 6
WORLD_SIZE = 1.0
COMM_RANGE = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPDATE_INTERVAL = 0.05

# Phase 1: SEI Configuration
SEI_MATCH_THRESHOLD = 0.85
TRIPLET_MARGIN = 0.2  # α from Algorithm 1
SNR_THRESHOLD_BASE = 0.8
SNR_ADAPTIVE_FACTOR = 0.15  # τ_adaptive from Algorithm 2

# Phase 2: PQC Configuration
PQC_KEY_SIZE_KB = 3.2  # Kyber public key
PQC_SIG_SIZE_KB = 2.4  # Dilithium signature
AES_IV_SIZE_KB = 0.016
AES_TAG_SIZE_KB = 0.016
NETWORK_RATE_MBPS = 25
BASE_NET_LATENCY_MS = 12
CRYPTO_MODES = ["NO_PQC", "HYBRID_PQC", "FULL_PQC"]

# Phase 3 & 4: Federated Learning Configuration
ENABLE_PHASE_3_ZERO_TRUST = True
MIN_TRUST_SCORE = 0.4
STRICTNESS_FACTOR = 1.5  # β from Algorithm 3
MAX_HISTORY_SIZE = 100

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
    'gbs': '#9400D3',
    'suspicious': '#FFA500',
    'byzantine': '#8B0000'
}

# =======================
# PHASE 1: SEI DEFENSE (Algorithm 1)
# =======================
class LSNetEmbeddingExtractor:
    """Implements LSNet feature extraction with triplet loss optimization"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = LSNet(embedding_dim=128).to(DEVICE)
        self.model.eval()
        self.anchor_db = {}  # Trusted reference database
        self.adaptive_threshold = SEI_MATCH_THRESHOLD
        
    @torch.no_grad()
    def extract_embedding(self, spectrogram: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 128-D embedding from spectrogram"""
        # Prepare input: spectrogram should be shape (H, W)
        if spectrogram.ndim == 2:
            # Convert to 2-channel (real/imag or magnitude/phase representation)
            spec_tensor = torch.tensor(spectrogram).float().to(DEVICE)
            # Duplicate to create 2 channels (LSNet expects 2 channels)
            spec_tensor = spec_tensor.unsqueeze(0).repeat(2, 1, 1).unsqueeze(0)
        else:
            spec_tensor = torch.tensor(spectrogram).float().to(DEVICE)
            if spec_tensor.dim() == 3:
                spec_tensor = spec_tensor.unsqueeze(0)
        
        # Forward pass
        embedding = self.model(spec_tensor)
        return embedding.cpu().numpy().squeeze()
    
    def compute_triplet_loss(self, anchor: np.ndarray, positive: np.ndarray, 
                            negative: np.ndarray, margin: float = TRIPLET_MARGIN) -> float:
        """Compute triplet loss: L = max(0, d_pos - d_neg + α)"""
        d_pos = np.linalg.norm(anchor - positive) ** 2
        d_neg = np.linalg.norm(anchor - negative) ** 2
        loss = max(0, d_pos - d_neg + margin)
        return loss
    
    def add_to_anchor_db(self, drone_id: str, embedding: np.ndarray, 
                         drone_type: str, snr: float):
        """Add trusted drone embedding to reference database"""
        self.anchor_db[drone_id] = {
            'embedding': embedding,
            'drone_type': drone_type,
            'snr': snr,
            'timestamp': time.time()
        }
    
    def verify_hardware(self, embedding: np.ndarray, snr: float) -> Tuple[bool, float, float]:
        """Verify hardware fingerprint against anchor database"""
        if not self.anchor_db:
            return False, 1.0, float('inf')
        
        # Find closest anchor
        min_distance = float('inf')
        closest_anchor = None
        
        for anchor_id, anchor_data in self.anchor_db.items():
            # L2 distance (Algorithm 1, lines 12-13)
            distance = np.linalg.norm(embedding - anchor_data['embedding'])
            if distance < min_distance:
                min_distance = distance
                closest_anchor = anchor_data
        
        # Adaptive threshold based on SNR (Gate 1)
        # Lower SNR → higher threshold (more tolerant)
        snr_factor = max(0.5, min(1.5, 15 / max(snr, 1)))
        adaptive_threshold = SEI_MATCH_THRESHOLD * snr_factor
        
        # Convert distance to similarity (smaller distance = higher similarity)
        similarity = 1.0 / (1.0 + min_distance)
        
        is_verified = similarity >= adaptive_threshold
        
        return is_verified, similarity, min_distance

# =======================
# PHASE 2: PQC HANDSHAKE (Algorithm 2)
# =======================
class KyberModule:
    """ML-KEM (Kyber) implementation simulation"""
    
    def __init__(self):
        self.public_key = None
        self.secret_key = None
        
    def generate_keys(self):
        """Generate Kyber keypair"""
        time.sleep(random.uniform(2, 4) / 1000)
        self.public_key = hashlib.sha256(b"kyber_pk").hexdigest()[:32]
        self.secret_key = hashlib.sha256(b"kyber_sk").hexdigest()[:32]
        return self.public_key, self.secret_key
    
    def encapsulate(self, public_key: str) -> Tuple[str, str]:
        """Kyber.Encapsulate: Generate ciphertext and shared secret"""
        time.sleep(random.uniform(3, 6) / 1000)
        ciphertext = hashlib.sha256(f"ct_{public_key}_{time.time()}".encode()).hexdigest()[:48]
        shared_secret = hashlib.sha256(f"ss_{public_key}_{time.time()}".encode()).hexdigest()[:32]
        return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: str, secret_key: str) -> str:
        """Kyber.Decapsulate: Recover shared secret"""
        time.sleep(random.uniform(2, 4) / 1000)
        shared_secret = hashlib.sha256(f"ss_{ciphertext}_{secret_key}".encode()).hexdigest()[:32]
        return shared_secret

class DilithiumModule:
    """ML-DSA (Dilithium) implementation simulation"""
    
    def __init__(self):
        self.public_key = None
        self.secret_key = None
        
    def generate_keys(self):
        """Generate Dilithium keypair"""
        time.sleep(random.uniform(3, 5) / 1000)
        self.public_key = hashlib.sha256(b"dilithium_pk").hexdigest()[:64]
        self.secret_key = hashlib.sha256(b"dilithium_sk").hexdigest()[:64]
        return self.public_key, self.secret_key
    
    def sign(self, message: str, secret_key: str) -> str:
        """Dilithium.Sign: Generate signature"""
        time.sleep(random.uniform(4, 8) / 1000)
        signature = hashlib.sha256(f"sig_{message}_{secret_key}".encode()).hexdigest()[:96]
        return signature
    
    def verify(self, message: str, signature: str, public_key: str) -> bool:
        """Dilithium.Verify: Verify signature"""
        time.sleep(random.uniform(3, 6) / 1000)
        expected = hashlib.sha256(f"sig_{message}_{public_key}".encode()).hexdigest()[:96]
        return signature == expected

class PQCHandshake:
    """Implements Algorithm 2: Zero-Trust Swarm Entry"""
    
    def __init__(self):
        self.kyber = KyberModule()
        self.dilithium = DilithiumModule()
        self.metrics = []
        self.active_sessions = {}
        
    def perform_handshake(self, drone_id: str, sei_embedding: np.ndarray, 
                          sei_verified: bool, mode: str = "HYBRID_PQC") -> Tuple[bool, Dict, float]:
        """
        Perform complete PQC handshake (Algorithm 2)
        
        Returns:
            success, session_info, total_latency_ms
        """
        start_time = time.time()
        
        if not sei_verified:
            # Gate 1 failed
            return False, {"error": "SEI verification failed"}, 0
        
        # Phase 1: Physical Challenge (already done by SEI)
        
        # Phase 2: Dilithium Authentication
        dilithium_start = time.time()
        pk_dilithium, sk_dilithium = self.dilithium.generate_keys()
        
        # Simulate signing and verification
        message = f"auth_request_{drone_id}_{time.time()}"
        signature = self.dilithium.sign(message, sk_dilithium)
        sig_valid = self.dilithium.verify(message, signature, pk_dilithium)
        
        if not sig_valid:
            return False, {"error": "Dilithium verification failed"}, (time.time() - start_time) * 1000
        
        dilithium_time = (time.time() - dilithium_start) * 1000
        
        # Phase 3: Kyber Key Exchange
        kyber_start = time.time()
        pk_kyber, sk_kyber = self.kyber.generate_keys()
        
        # Encapsulate to generate shared secret
        ciphertext, shared_secret = self.kyber.encapsulate(pk_kyber)
        
        # Derive session key using KDF
        nonce = hashlib.sha256(f"nonce_{drone_id}_{time.time()}".encode()).digest()[:16]
        session_key = hashlib.sha256(f"{shared_secret}{nonce}".encode()).hexdigest()[:32]
        
        kyber_time = (time.time() - kyber_start) * 1000
        
        # Store session
        self.active_sessions[drone_id] = {
            'session_key': session_key,
            'ciphertext': ciphertext,
            'nonce': nonce.hex(),
            'timestamp': time.time(),
            'mode': mode
        }
        
        total_latency = (time.time() - start_time) * 1000
        
        # Calculate metrics
        metric = {
            'mode': mode,
            'drone_id': drone_id,
            'L_total': total_latency,
            'L_dilithium': dilithium_time,
            'L_kyber': kyber_time,
            'L_crypto': dilithium_time + kyber_time,
            'B_pqc': PQC_KEY_SIZE_KB + PQC_SIG_SIZE_KB,
            'B_aes': AES_IV_SIZE_KB + AES_TAG_SIZE_KB,
            'session_key': session_key[:8] + "..."
        }
        
        self.metrics.append(metric)
        
        return True, metric, total_latency
    
    def encrypt_payload(self, drone_id: str, payload: Dict, mode: str = "HYBRID_PQC") -> Tuple[Dict, Dict]:
        """Encrypt payload using established session key"""
        if drone_id not in self.active_sessions:
            return payload, {"error": "No active session"}
        
        session = self.active_sessions[drone_id]
        
        # Simulate AES-256-GCM encryption
        start = time.time()
        
        # Add encryption overhead
        if mode == "HYBRID_PQC":
            crypto_time = random.uniform(0.2, 0.6)
        elif mode == "FULL_PQC":
            crypto_time = random.uniform(0.5, 1.2)
        else:
            crypto_time = 0
            
        time.sleep(crypto_time / 1000)
        
        # Calculate bandwidth overhead
        if mode == "NO_PQC":
            bandwidth = 0
        elif mode == "HYBRID_PQC":
            bandwidth = AES_IV_SIZE_KB + AES_TAG_SIZE_KB
        else:  # FULL_PQC
            bandwidth = PQC_KEY_SIZE_KB + PQC_SIG_SIZE_KB + AES_IV_SIZE_KB + AES_TAG_SIZE_KB
        
        encrypted_payload = payload.copy()
        encrypted_payload['encrypted'] = True
        encrypted_payload['session_id'] = drone_id
        
        metric = {
            'mode': mode,
            'L_crypto': crypto_time,
            'B_overhead': bandwidth,
            'timestamp': time.time()
        }
        
        return encrypted_payload, metric


# =======================
# PHASE 3-4: FEDERATED LEARNING WITH ZERO-TRUST (Algorithm 3)
# =======================
class FederatedLearningNode:
    """Implements Byzantine-robust federated learning with zero-trust"""
    
    def __init__(self, node_id: str, is_gbs: bool = False):
        self.node_id = node_id
        self.is_gbs = is_gbs  # Ground Base Station
        self.local_model = None
        self.trust_score = 1.0 if is_gbs else 0.5
        self.history = deque(maxlen=MAX_HISTORY_SIZE)
        self.contributions = []
        self.byzantine_count = 0
        self.suspicious_count = 0
        
    def compute_local_gradient(self, data_batch):
        """Simulate local gradient computation"""
        # Random gradient for simulation
        gradient_size = random.randint(1000, 5000)
        gradient = np.random.randn(gradient_size) * 0.01
        
        # Simulate computation time
        time.sleep(random.uniform(5, 15) / 1000)
        
        # Byzantine nodes send malicious gradients
        if self.trust_score < MIN_TRUST_SCORE and random.random() > 0.3:
            # Malicious behavior: flip gradient direction
            gradient = -gradient * random.uniform(0.5, 2.0)
            self.byzantine_count += 1
            
        return gradient
    
    def update_trust_score(self, delta: float):
        """Update trust score based on contribution quality"""
        old_score = self.trust_score
        self.trust_score = max(0.0, min(1.0, self.trust_score + delta))
        
        if delta < -0.1:
            self.suspicious_count += 1
            
        self.history.append({
            'timestamp': time.time(),
            'old_score': old_score,
            'new_score': self.trust_score,
            'delta': delta
        })
        
        return self.trust_score

class ZeroTrustAggregator:
    """Implements Algorithm 3: Zero-Trust Federated Aggregation"""
    
    def __init__(self):
        self.nodes = {}  # node_id -> FederatedLearningNode
        self.global_model = None
        self.aggregation_round = 0
        self.anomaly_log = []
        
    def register_node(self, node_id: str, is_gbs: bool = False):
        """Register a new node in the FL network"""
        self.nodes[node_id] = FederatedLearningNode(node_id, is_gbs)
        return self.nodes[node_id]
    
    def compute_trust_weight(self, trust_score: float, 
                            strictness: float = STRICTNESS_FACTOR) -> float:
        """Compute aggregation weight based on trust score (Equation 9)"""
        if trust_score >= MIN_TRUST_SCORE:
            # Trusted nodes get weight proportional to trust score
            return (trust_score - MIN_TRUST_SCORE) / (1 - MIN_TRUST_SCORE)
        else:
            # Untrusted nodes get reduced weight (β times lower)
            return max(0, (trust_score / MIN_TRUST_SCORE - 1) * strictness)
    
    def detect_byzantine(self, gradients: List[np.ndarray], 
                        trust_scores: List[float]) -> List[int]:
        """Detect Byzantine nodes using Krum-like algorithm"""
        n_nodes = len(gradients)
        if n_nodes < 3:
            return []
        
        # Compute pairwise distances between gradients
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Cosine distance weighted by trust scores
                cos_dist = 1 - np.dot(gradients[i], gradients[j]) / (
                    np.linalg.norm(gradients[i]) * np.linalg.norm(gradients[j]) + 1e-8)
                
                trust_factor = (trust_scores[i] + trust_scores[j]) / 2
                distances[i, j] = cos_dist / (trust_factor + 1e-8)
                distances[j, i] = distances[i, j]
        
        # For each node, sum distances to closest k nodes
        k = max(1, n_nodes - 2)  # Exclude at most 2 nodes
        node_scores = []
        
        for i in range(n_nodes):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:k])
            node_scores.append((i, score))
        
        # Sort by score (lower is better)
        node_scores.sort(key=lambda x: x[1])
        
        # Bottom k+1 nodes are considered trusted
        trusted_indices = [idx for idx, _ in node_scores[:k+1]]
        suspicious_indices = [idx for idx, _ in node_scores[k+1:]]
        
        return suspicious_indices
    
    def aggregate_round(self, contributions: Dict[str, np.ndarray]) -> Dict:
        """
        Perform one round of zero-trust aggregation (Algorithm 3)
        
        Args:
            contributions: node_id -> gradient
            
        Returns:
            aggregated_gradient and metrics
        """
        self.aggregation_round += 1
        
        # Prepare lists
        node_ids = list(contributions.keys())
        gradients = [contributions[nid] for nid in node_ids]
        trust_scores = [self.nodes[nid].trust_score for nid in node_ids]
        
        # Step 1: Detect Byzantine nodes (lines 1-2)
        suspicious_indices = self.detect_byzantine(gradients, trust_scores)
        suspicious_nodes = [node_ids[i] for i in suspicious_indices]
        
        # Step 2: Update trust scores for suspicious nodes (lines 3-6)
        for node_id in suspicious_nodes:
            delta = -0.1 * (1 - self.nodes[node_id].trust_score)
            self.nodes[node_id].update_trust_score(delta)
            self.anomaly_log.append({
                'round': self.aggregation_round,
                'node_id': node_id,
                'event': 'suspicious_activity',
                'delta': delta
            })
        
        # Step 3: Compute trust-weighted aggregation (lines 7-9)
        weights = []
        filtered_gradients = []
        filtered_node_ids = []
        
        for i, node_id in enumerate(node_ids):
            trust_score = self.nodes[node_id].trust_score
            
            # Filter out nodes with trust score below threshold
            if trust_score >= MIN_TRUST_SCORE:
                weight = self.compute_trust_weight(trust_score)
                weights.append(weight)
                filtered_gradients.append(gradients[i])
                filtered_node_ids.append(node_id)
            else:
                # Very low trust nodes get penalized more
                delta = -0.2 * (MIN_TRUST_SCORE - trust_score)
                self.nodes[node_id].update_trust_score(delta)
                self.anomaly_log.append({
                    'round': self.aggregation_round,
                    'node_id': node_id,
                    'event': 'low_trust_exclusion',
                    'delta': delta
                })
        
        if not filtered_gradients:
            # No trusted nodes, return None (aggregation fails)
            return {
                'success': False,
                'aggregated': None,
                'trusted_count': 0,
                'total_count': len(node_ids),
                'suspicious_nodes': suspicious_nodes
            }
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted average of gradients
        aggregated = np.zeros_like(filtered_gradients[0])
        for i, grad in enumerate(filtered_gradients):
            aggregated += weights[i] * grad
        
        # Update trust scores for contributing nodes (positive reinforcement)
        for node_id in filtered_node_ids:
            delta = 0.05 * weights[filtered_node_ids.index(node_id)]
            self.nodes[node_id].update_trust_score(delta)
        
        # Step 4: Generate metrics
        metrics = {
            'round': self.aggregation_round,
            'success': True,
            'trusted_count': len(filtered_node_ids),
            'total_count': len(node_ids),
            'suspicious_count': len(suspicious_nodes),
            'avg_trust_score': np.mean([self.nodes[nid].trust_score for nid in node_ids]),
            'trusted_nodes': filtered_node_ids,
            'suspicious_nodes': suspicious_nodes
        }
        
        return {
            'success': True,
            'aggregated': aggregated,
            'metrics': metrics,
            'weights': dict(zip(filtered_node_ids, weights))
        }
# =======================
# UAV SWARM SIMULATION
# =======================
class UAVDrone:
    """Represents a single UAV drone in the swarm"""
    
    def __init__(self, drone_id: int, position: np.ndarray, 
                 drone_type: str = "friendly", 
                 sei_extractor: LSNetEmbeddingExtractor = None):
        self.drone_id = drone_id
        self.position = position.astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.drone_type = drone_type
        self.role = "unknown"
        self.trust_status = "unknown"  # trusted, malicious, suspicious, isolated
        self.trust_score = 0.5
        self.snr = random.uniform(5, 25)  # dB
        self.active = True
        self.visible = True
        
        # SEI fingerprint
        self.sei_embedding = self._generate_sei_fingerprint() if sei_extractor else None
        
        # PQC session
        self.pqc_session = None
        self.session_key = None
        self.handshake_complete = False
        
        # FL node
        self.fl_node = None
        
        # Movement
        self.target_position = None
        self.speed = random.uniform(0.01, 0.05)
        self.path_history = deque(maxlen=50)
        self.path_history.append(position.copy())
        
        # Communication
        self.neighbors = []
        self.messages_sent = 0
        self.messages_received = 0
        
        # For Byzantine behavior simulation
        self.byzantine_pattern = random.choice(['flip_gradient', 'random_noise', 'constant', 'none'])
        self.malicious_behavior_prob = 0.0
        
        # Set role based on type
        if drone_type == "gbs":
            self.role = "gbs"
            self.trust_status = "trusted"
            self.trust_score = 1.0
        elif drone_type == "friendly":
            self.role = "scout"
            self.trust_status = "trusted"
            self.trust_score = random.uniform(0.7, 0.95)
        elif drone_type == "enemy":
            self.role = "enemy"
            self.trust_status = "malicious"
            self.trust_score = random.uniform(0.1, 0.3)
            self.malicious_behavior_prob = 0.8
        elif drone_type == "byzantine":
            self.role = "byzantine"
            self.trust_status = "suspicious"
            self.trust_score = random.uniform(0.2, 0.5)
            self.malicious_behavior_prob = 0.6
        else:  # unknown
            self.role = "unknown"
            self.trust_status = "unknown"
            self.trust_score = random.uniform(0.3, 0.6)
            
    def _generate_sei_fingerprint(self) -> np.ndarray:
        """Generate simulated SEI fingerprint"""
        # Generate random 128-dim normalized embedding
        embedding = np.random.randn(128)
        # Add device-specific variations based on drone type
        if self.drone_type == "friendly":
            # Friendly drones have consistent fingerprints
            base = np.random.randn(128) * 0.1
            embedding = embedding + base
        elif self.drone_type == "enemy":
            # Enemy drones have different fingerprint pattern
            embedding = embedding * 1.5 + np.random.randn(128) * 0.3
        elif self.drone_type == "byzantine":
            # Byzantine nodes might spoof but with variations
            embedding = embedding + np.random.randn(128) * 0.2
            
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def update_position(self):
        """Update drone position with simple movement"""
        if not self.active:
            return
            
        # Random walk with some persistence
        if self.target_position is None or np.linalg.norm(self.target_position - self.position) < 0.05:
            # Set new random target within world bounds
            self.target_position = np.random.rand(2) * WORLD_SIZE
        
        # Move towards target
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            self.velocity = (direction / distance) * self.speed
            self.position += self.velocity
            
            # Keep within bounds
            self.position = np.clip(self.position, 0, WORLD_SIZE)
            
        # Record path
        self.path_history.append(self.position.copy())
        
    def find_neighbors(self, all_drones):
        """Find neighboring drones within communication range"""
        self.neighbors = []
        for drone in all_drones:
            if drone.drone_id != self.drone_id and drone.active:
                distance = np.linalg.norm(self.position - drone.position)
                if distance <= COMM_RANGE:
                    self.neighbors.append(drone.drone_id)
        return self.neighbors
    
    def perform_handshake(self, pqc_handshake: PQCHandshake, 
                         sei_extractor: LSNetEmbeddingExtractor,
                         mode: str = "HYBRID_PQC") -> Tuple[bool, Dict]:
        """Perform PQC handshake with this drone"""
        if self.handshake_complete:
            return True, {"message": "Handshake already complete"}
        
        # Step 1: SEI verification
        is_verified, similarity, distance = sei_extractor.verify_hardware(
            self.sei_embedding, self.snr
        )
        
        # Step 2: PQC handshake
        success, session_info, latency = pqc_handshake.perform_handshake(
            f"drone_{self.drone_id}", self.sei_embedding, is_verified, mode
        )
        
        if success:
            self.pqc_session = pqc_handshake
            self.session_key = session_info.get('session_key') if session_info else None
            self.handshake_complete = True
            
            # Update trust status based on handshake
            if is_verified:
                self.trust_status = "trusted"
                self.trust_score = min(1.0, self.trust_score + 0.1)
            else:
                self.trust_status = "suspicious"
                self.trust_score = max(0.0, self.trust_score - 0.15)
        
        return success, session_info
    
    def to_dict(self):
        """Convert drone to dictionary for visualization"""
        return {
            'id': self.drone_id,
            'x': float(self.position[0]),
            'y': float(self.position[1]),
            'type': self.drone_type,
            'role': self.role,
            'trust_status': self.trust_status,
            'trust_score': self.trust_score,
            'active': self.active,
            'visible': self.visible,
            'handshake_complete': self.handshake_complete,
            'neighbors': self.neighbors
        }

# =======================
# SIMULATION MANAGER
# =======================
class SwarmSimulation:
    """Main simulation manager integrating all phases"""
    
    def __init__(self):
        self.drones = []
        self.gbs_stations = []
        self.sei_extractor = LSNetEmbeddingExtractor()
        self.pqc_handshake = PQCHandshake()
        self.fl_aggregator = ZeroTrustAggregator()
        self.time = 0.0
        self.simulation_active = False
        self.selected_metrics = []
        
        # Statistics
        self.stats = {
            'trusted_count': [],
            'suspicious_count': [],
            'malicious_count': [],
            'handshake_success_rate': [],
            'avg_trust_score': [],
            'fl_rounds': 0,
            'detected_attacks': 0,
            'false_positives': 0,
            'latencies': [],
            'bandwidth_used': []
        }
        
        # Initialize with default drones
        self._initialize_drones()
        
    def _initialize_drones(self):
        """Initialize the swarm with drones"""
        # Add GBS (ground base stations) - these are always trusted
        gbs_positions = [
            np.array([0.1, 0.1]),
            np.array([0.9, 0.9]),
            np.array([0.5, 0.5])
        ]
        
        for i, pos in enumerate(gbs_positions):
            drone = UAVDrone(
                drone_id=i,
                position=pos,
                drone_type="gbs",
                sei_extractor=self.sei_extractor
            )
            self.drones.append(drone)
            self.gbs_stations.append(drone)
            self.fl_aggregator.register_node(f"drone_{i}", is_gbs=True)
            
        # Add friendly drones
        for i in range(len(gbs_positions), INITIAL_DRONES):
            pos = np.random.rand(2) * WORLD_SIZE
            drone_type = random.choices(
                ["friendly", "enemy", "byzantine"],
                weights=[0.6, 0.2, 0.2]
            )[0]
            
            drone = UAVDrone(
                drone_id=i,
                position=pos,
                drone_type=drone_type,
                sei_extractor=self.sei_extractor
            )
            self.drones.append(drone)
            self.fl_aggregator.register_node(f"drone_{i}")
            
    def step(self):
        """Advance simulation by one time step"""
        self.time += UPDATE_INTERVAL
        
        # Update drone positions
        for drone in self.drones:
            if drone.active:
                drone.update_position()
                
        # Update neighbor relationships
        for drone in self.drones:
            if drone.active:
                drone.find_neighbors(self.drones)
                
        # Periodic handshake checks for new drones
        if self.time % 1.0 < UPDATE_INTERVAL:  # Every second
            self._perform_periodic_checks()
            
        # Update statistics
        self._update_stats()
        
    def _perform_periodic_checks(self):
        """Perform periodic security checks"""
        trusted = sum(1 for d in self.drones if d.trust_status == "trusted" and d.active)
        suspicious = sum(1 for d in self.drones if d.trust_status == "suspicious" and d.active)
        malicious = sum(1 for d in self.drones if d.trust_status == "malicious" and d.active)
        
        self.stats['trusted_count'].append(trusted)
        self.stats['suspicious_count'].append(suspicious)
        self.stats['malicious_count'].append(malicious)
        
    def _update_stats(self):
        """Update simulation statistics"""
        active_drones = [d for d in self.drones if d.active]
        if active_drones:
            avg_trust = np.mean([d.trust_score for d in active_drones])
            self.stats['avg_trust_score'].append(avg_trust)
            
    def add_drone(self, drone_type: str = "friendly"):
        """Add a new drone to the swarm"""
        if len(self.drones) >= MAX_DRONES:
            return None
            
        new_id = len(self.drones)
        pos = np.random.rand(2) * WORLD_SIZE
        
        drone = UAVDrone(
            drone_id=new_id,
            position=pos,
            drone_type=drone_type,
            sei_extractor=self.sei_extractor
        )
        
        self.drones.append(drone)
        self.fl_aggregator.register_node(f"drone_{new_id}")
        
        return drone
        
    def remove_drone(self, drone_id: int):
        """Remove a drone from the swarm"""
        for drone in self.drones:
            if drone.drone_id == drone_id:
                drone.active = False
                return True
        return False
        
    def get_metrics(self):
        """Get current simulation metrics"""
        active_drones = [d for d in self.drones if d.active]
        
        metrics = {
            'time': self.time,
            'total_drones': len(active_drones),
            'trusted': sum(1 for d in active_drones if d.trust_status == "trusted"),
            'suspicious': sum(1 for d in active_drones if d.trust_status == "suspicious"),
            'malicious': sum(1 for d in active_drones if d.trust_status == "malicious"),
            'unknown': sum(1 for d in active_drones if d.trust_status == "unknown"),
            'avg_trust_score': np.mean([d.trust_score for d in active_drones]) if active_drones else 0,
            'handshake_complete': sum(1 for d in active_drones if d.handshake_complete),
            'fl_rounds': self.fl_aggregator.aggregation_round,
            'detected_attacks': self.stats['detected_attacks'][-1] if self.stats['detected_attacks'] else 0
        }
        
        return metrics

# =======================
# VISUALIZATION FUNCTIONS
# =======================
def create_swarm_visualization(simulation: SwarmSimulation, width: int = 800, height: int = 600):
    """Create matplotlib visualization of the swarm"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot communication links
    for drone in simulation.drones:
        if drone.active and drone.visible:
            for neighbor_id in drone.neighbors[:5]:  # Limit to avoid clutter
                neighbor = simulation.drones[neighbor_id]
                if neighbor.active:
                    # Determine link color based on trust
                    if drone.trust_status == "trusted" and neighbor.trust_status == "trusted":
                        color = COLORS['communication']
                        alpha = 0.3
                    elif "malicious" in [drone.trust_status, neighbor.trust_status]:
                        color = COLORS['malicious']
                        alpha = 0.2
                    else:
                        color = COLORS['suspicious']
                        alpha = 0.15
                        
                    ax.plot([drone.position[0], neighbor.position[0]],
                           [drone.position[1], neighbor.position[1]],
                           color=color, alpha=alpha, linewidth=1, zorder=1)
    
    # Plot drones
    for drone in simulation.drones:
        if not drone.active or not drone.visible:
            continue
            
        # Determine color based on type and trust status
        if drone.drone_type == "gbs":
            color = COLORS['gbs']
            marker = 's'
            size = 200
        elif drone.trust_status == "trusted":
            color = COLORS['trusted']
            marker = 'o'
            size = 100
        elif drone.trust_status == "malicious":
            color = COLORS['malicious']
            marker = 'X'
            size = 120
        elif drone.trust_status == "suspicious":
            color = COLORS['suspicious']
            marker = '^'
            size = 100
        else:
            color = COLORS['unknown']
            marker = 'o'
            size = 80
            
        # Plot drone
        ax.scatter(drone.position[0], drone.position[1], 
                  c=color, s=size, marker=marker, 
                  edgecolors='white', linewidth=1, zorder=2,
                  label=f"{drone.drone_type}" if drone.drone_id == 0 else "")
        
        # Add drone ID
        ax.annotate(str(drone.drone_id), 
                   (drone.position[0], drone.position[1]),
                   color='white', fontsize=8, ha='center', va='center',
                   weight='bold', zorder=3)
        
        # Draw path trail
        if len(drone.path_history) > 1:
            path = np.array(drone.path_history)
            ax.plot(path[:, 0], path[:, 1], 
                   color=color, alpha=0.3, linewidth=1, zorder=0)
    
    # Configure plot
    ax.set_xlim(-0.05, WORLD_SIZE + 0.05)
    ax.set_ylim(-0.05, WORLD_SIZE + 0.05)
    ax.set_title(f"UAV Swarm State - Time: {simulation.time:.1f}s", 
                color=COLORS['text'], fontsize=14, pad=20)
    ax.set_xlabel("X Position", color=COLORS['text'])
    ax.set_ylabel("Y Position", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Trusted',
                  markerfacecolor=COLORS['trusted'], markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', label='Suspicious',
                  markerfacecolor=COLORS['suspicious'], markersize=10),
        plt.Line2D([0], [0], marker='X', color='w', label='Malicious',
                  markerfacecolor=COLORS['malicious'], markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='GBS',
                  markerfacecolor=COLORS['gbs'], markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             facecolor=COLORS['background'], labelcolor=COLORS['text'])
    
    plt.tight_layout()
    return fig

def create_metrics_visualization(simulation: SwarmSimulation):
    """Create metrics dashboard visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=COLORS['background'])
    
    # 1. Trust distribution over time
    ax = axes[0, 0]
    if simulation.stats['trusted_count']:
        time_steps = np.arange(len(simulation.stats['trusted_count'])) * UPDATE_INTERVAL
        ax.plot(time_steps, simulation.stats['trusted_count'], 
                color=COLORS['trusted'], label='Trusted', linewidth=2)
        ax.plot(time_steps, simulation.stats['suspicious_count'], 
                color=COLORS['suspicious'], label='Suspicious', linewidth=2)
        ax.plot(time_steps, simulation.stats['malicious_count'], 
                color=COLORS['malicious'], label='Malicious', linewidth=2)
    ax.set_title("Drone Trust Distribution", color=COLORS['text'])
    ax.set_xlabel("Time (s)", color=COLORS['text'])
    ax.set_ylabel("Count", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.legend(facecolor=COLORS['background'], labelcolor=COLORS['text'])
    ax.grid(True, alpha=0.3)
    
    # 2. Average trust score
    ax = axes[0, 1]
    if simulation.stats['avg_trust_score']:
        time_steps = np.arange(len(simulation.stats['avg_trust_score'])) * UPDATE_INTERVAL
        ax.plot(time_steps, simulation.stats['avg_trust_score'], 
                color='cyan', linewidth=2)
        ax.axhline(y=MIN_TRUST_SCORE, color='red', linestyle='--', 
                   label=f'Min Trust ({MIN_TRUST_SCORE})')
    ax.set_title("Average Trust Score", color=COLORS['text'])
    ax.set_xlabel("Time (s)", color=COLORS['text'])
    ax.set_ylabel("Trust Score", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. Handshake success rate
    ax = axes[0, 2]
    if simulation.stats['handshake_success_rate']:
        ax.plot(simulation.stats['handshake_success_rate'], 
                color='orange', linewidth=2)
    ax.set_title("Handshake Success Rate", color=COLORS['text'])
    ax.set_xlabel("Attempts", color=COLORS['text'])
    ax.set_ylabel("Success Rate", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Latency distribution
    ax = axes[1, 0]
    if simulation.stats['latencies']:
        ax.hist(simulation.stats['latencies'], bins=20, 
                color='skyblue', edgecolor='white')
    ax.set_title("Handshake Latency Distribution", color=COLORS['text'])
    ax.set_xlabel("Latency (ms)", color=COLORS['text'])
    ax.set_ylabel("Frequency", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, alpha=0.3)
    
    # 5. FL rounds and detections
    ax = axes[1, 1]
    fl_rounds = simulation.stats['fl_rounds']
    detections = simulation.stats['detected_attacks'][-1] if simulation.stats['detected_attacks'] else 0
    false_positives = simulation.stats['false_positives'][-1] if simulation.stats['false_positives'] else 0
    
    bars = ax.bar(['FL Rounds', 'Detected', 'False Pos.'], 
                  [fl_rounds, detections, false_positives],
                  color=['green', 'red', 'orange'])
    ax.set_title("Federated Learning Stats", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    
    # 6. Node trust scores
    ax = axes[1, 2]
    active_drones = [d for d in simulation.drones if d.active]
    if active_drones:
        drone_ids = [d.drone_id for d in active_drones]
        trust_scores = [d.trust_score for d in active_drones]
        colors_list = [COLORS['trusted'] if d.trust_status == "trusted" 
                      else COLORS['suspicious'] if d.trust_status == "suspicious"
                      else COLORS['malicious'] for d in active_drones]
        
        ax.bar(drone_ids, trust_scores, color=colors_list)
        ax.axhline(y=MIN_TRUST_SCORE, color='red', linestyle='--', 
                   label=f'Min Trust')
    ax.set_title("Current Node Trust Scores", color=COLORS['text'])
    ax.set_xlabel("Drone ID", color=COLORS['text'])
    ax.set_ylabel("Trust Score", color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
# =======================
# STREAMLIT UI
# =======================
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'simulation' not in st.session_state:
        st.session_state.simulation = SwarmSimulation()
        st.session_state.simulation_active = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 0.5  # seconds
        st.session_state.crypto_mode = "HYBRID_PQC"
        st.session_state.show_metrics = True
        st.session_state.selected_drone = None
        st.session_state.event_log = []
        
def add_to_log(message: str, level: str = "INFO"):
    """Add message to event log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.append(f"[{timestamp}] [{level}] {message}")
    # Keep only last 100 messages
    if len(st.session_state.event_log) > 100:
        st.session_state.event_log = st.session_state.event_log[-100:]

def main():
    """Main Streamlit application"""
    st.title("🚁 Q-SEI-FED: Quantum-Secure UAV Swarm with Zero-Trust Federated Learning")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar for controls
    with st.sidebar:
        st.header("🎮 Simulation Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.simulation_active = True
                add_to_log("Simulation started")
                
        with col2:
            if st.button("⏸️ Stop", use_container_width=True):
                st.session_state.simulation_active = False
                add_to_log("Simulation stopped")
                
        st.divider()
        
        # Drone management
        st.subheader("🚁 Drone Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            drone_type = st.selectbox("Type", ["friendly", "enemy", "byzantine", "gbs"])
        with col2:
            if st.button("➕ Add", use_container_width=True):
                new_drone = st.session_state.simulation.add_drone(drone_type)
                if new_drone:
                    add_to_log(f"Added new {drone_type} drone (ID: {new_drone.drone_id})")
                else:
                    st.warning("Max drones reached!")
                    
        with col3:
            if st.button("➖ Remove", use_container_width=True):
                if st.session_state.selected_drone is not None:
                    st.session_state.simulation.remove_drone(st.session_state.selected_drone)
                    add_to_log(f"Removed drone {st.session_state.selected_drone}")
                    st.session_state.selected_drone = None
                    
        # Configuration
        st.divider()
        st.subheader("⚙️ Configuration")
        
        st.session_state.crypto_mode = st.selectbox(
            "Crypto Mode",
            CRYPTO_MODES,
            index=1
        )
        
        st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=True)
        if st.session_state.auto_refresh:
            st.session_state.refresh_rate = st.slider(
                "Refresh rate (s)", 0.1, 2.0, 0.5, 0.1
            )
            
        # Metrics selection
        st.divider()
        st.subheader("📊 Metrics")
        show_trust = st.checkbox("Trust Distribution", value=True)
        show_latency = st.checkbox("Latency", value=True)
        show_fl = st.checkbox("FL Statistics", value=True)
        
        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.session_state.simulation = SwarmSimulation()
            st.session_state.event_log = []
            add_to_log("Simulation reset")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualization area
        viz_placeholder = st.empty()
        
        # Metrics area
        metrics_placeholder = st.empty()
        
    with col2:
        # Drone info panel
        st.subheader("🎯 Drone Information")
        
        # Drone selector
        active_drones = [d for d in st.session_state.simulation.drones if d.active]
        drone_options = {f"Drone {d.drone_id} ({d.drone_type})": d.drone_id for d in active_drones}
        
        if drone_options:
            selected = st.selectbox("Select Drone", list(drone_options.keys()))
            st.session_state.selected_drone = drone_options[selected]
            
            # Show drone details
            if st.session_state.selected_drone is not None:
                drone = next((d for d in active_drones if d.drone_id == st.session_state.selected_drone), None)
                if drone:
                    with st.container(border=True):
                        st.write(f"**ID:** {drone.drone_id}")
                        st.write(f"**Type:** {drone.drone_type}")
                        st.write(f"**Role:** {drone.role}")
                        st.write(f"**Status:** {drone.trust_status}")
                        st.write(f"**Trust Score:** {drone.trust_score:.3f}")
                        st.write(f"**SNR:** {drone.snr:.1f} dB")
                        st.write(f"**Position:** ({drone.position[0]:.3f}, {drone.position[1]:.3f})")
                        st.write(f"**Handshake:** {'✓' if drone.handshake_complete else '✗'}")
                        st.write(f"**Neighbors:** {len(drone.neighbors)}")
        else:
            st.info("No active drones")
            
        # Event log
        st.divider()
        st.subheader("📋 Event Log")
        log_container = st.container(height=300)
        with log_container:
            for log in st.session_state.event_log[-20:]:
                st.text(log)
                
    # Simulation step
    if st.session_state.simulation_active:
        # Perform simulation step
        st.session_state.simulation.step()
        
        # Generate visualizations
        swarm_fig = create_swarm_visualization(st.session_state.simulation)
        metrics_fig = create_metrics_visualization(st.session_state.simulation)
        
        # Update placeholders
        with viz_placeholder.container():
            st.pyplot(swarm_fig)
            
        with metrics_placeholder.container():
            st.pyplot(metrics_fig)
            
        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_rate)
            st.rerun()
            
    else:
        # Show static visualization when paused
        swarm_fig = create_swarm_visualization(st.session_state.simulation)
        metrics_fig = create_metrics_visualization(st.session_state.simulation)
        
        with viz_placeholder.container():
            st.pyplot(swarm_fig)
            
        with metrics_placeholder.container():
            st.pyplot(metrics_fig)
            
    # Footer with system info
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = st.session_state.simulation.get_metrics()
    
    with col1:
        st.metric("Total Drones", metrics['total_drones'])
    with col2:
        st.metric("Trusted", metrics['trusted'])
    with col3:
        st.metric("Suspicious", metrics['suspicious'])
    with col4:
        st.metric("Malicious", metrics['malicious'])
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Trust", f"{metrics['avg_trust_score']:.3f}")
    with col2:
        st.metric("FL Rounds", metrics['fl_rounds'])
    with col3:
        st.metric("Handshake Complete", metrics['handshake_complete'])
    with col4:
        st.metric("Detected Attacks", metrics['detected_attacks'])

if __name__ == "__main__":
    main()