"""
src/config.py
Configuration file for ECG Arrhythmia Classification (PhysioNet 2017 Challenge).
Centralizes paths, signal processing parameters, training hyperparameters, and class mappings.
"""

from pathlib import Path
import torch

# ==========================================
# 1. Project Directory Structure
# ==========================================
# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Reference & Split CSV Files
REFERENCE_CSV_PATH = RAW_DATA_DIR / "REFERENCE.csv"
TRAIN_SPLIT_CSV = SPLITS_DIR / "train.csv"
VAL_SPLIT_CSV = SPLITS_DIR / "val.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test.csv"

# Aliases for backward compatibility
TRAIN_CSV = TRAIN_SPLIT_CSV
VAL_CSV = VAL_SPLIT_CSV
TEST_CSV = TEST_SPLIT_CSV

# Output / Experiments Paths
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# Ensure essential output directories exist
for directory in [CHECKPOINT_DIR, LOGS_DIR, RESULTS_DIR, FIGURES_DIR, SPLITS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Signal Processing Parameters
# ==========================================
SAMPLING_RATE = 300  # PhysioNet 2017 sampling frequency in Hz
SIGNAL_DURATION = 30  # Target duration in seconds
TARGET_LENGTH = SAMPLING_RATE * SIGNAL_DURATION  # Fixed length: 9000 samples

# Filter parameters
APPLY_FILTERING = True
BANDPASS_LOWCUT = 0.5   # Hz (removes baseline wander)
BANDPASS_HIGHCUT = 45.0 # Hz (removes high-frequency noise & muscle artifacts)
BANDPASS_ORDER = 3      # Butterworth filter order
NOTCH_FREQ = 50.0       # Powerline interference frequency (Hz)
NOTCH_Q = 30.0          # Quality factor for Notch filter

# Normalization
NORMALIZATION_METHOD = "z_score"  # Options: 'z_score', 'min_max', None

# Training-time augmentation (applied after normalize + length adjust)
USE_AUGMENTATION = True
AUG_PROB = 0.5
AUG_NOISE_STD = 0.05
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_BASELINE_AMP = 0.1
AUG_TIME_STRETCH_MIN = 0.9
AUG_TIME_STRETCH_MAX = 1.1

# Test-time augmentation (sliding-window logit averaging)
TTA_WINDOW_LENGTH = TARGET_LENGTH
TTA_STRIDE = SAMPLING_RATE * 15  # 15 s overlap step (50% for 30 s windows)

# ==========================================
# 3. Class Definitions & Mappings
# ==========================================
CLASSES = ["N", "A", "O", "~"]
NUM_CLASSES = len(CLASSES)

LABEL_TO_IDX = {
    "N": 0,  # Normal rhythm
    "A": 1,  # Atrial Fibrillation (AF)
    "O": 2,  # Other rhythm
    "~": 3   # Noisy recording
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

CLASS_NAMES = {
    "N": "Normal",
    "A": "Atrial Fibrillation",
    "O": "Other Rhythm",
    "~": "Noisy"
}

# ==========================================
# 4. Training Hyperparameters (Defaults)
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 is safest on Windows to prevent multiprocessing freeze
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40
NUM_EPOCHS = EPOCHS
EARLY_STOPPING_PATIENCE = 10

# Cosine Annealing LR
USE_COSINE_LR = True
MIN_LR = 1e-6

# Class weights for Cross-Entropy Loss
CLASS_WEIGHTS = torch.tensor([0.42, 2.80, 0.88, 7.55], dtype=torch.float32)
