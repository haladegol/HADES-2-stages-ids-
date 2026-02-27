"""
Configuration file for HADES - Hierarchical Intrusion Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, DASHBOARD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_PATH = DATA_DIR / "CSE-CIC-IDS2018"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Database configuration
DB_PATH = LOGS_DIR / "hades_logs.db"

# Model configuration
STAGE1_MODEL_PATH = MODELS_DIR / "stage1_random_forest.pkl"
STAGE2_MODEL_PATH = MODELS_DIR / "stage2_xgboost.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = MODELS_DIR / "feature_lists.pkl"

# Model parameters
STAGE1_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 42
}

STAGE2_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

# Feature sets
STAGE1_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Fwd Header Length',
    'Bwd Header Length'
]

STAGE2_FEATURES = STAGE1_FEATURES + [
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'Fwd IAT Total',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init Fwd Win Bytes',
    'Init Bwd Win Bytes',
    'Fwd Act Data Packets',
    'Fwd Seg Size Min',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min'
]

# Attack categories mapping to MITRE ATT&CK
ATTACK_MAPPING = {
    'Benign': 'Benign',
    'DoS attacks-GoldenEye': 'DoS: GoldenEye',
    'DoS attacks-Slowloris': 'DoS: Slowloris',
    'DoS attacks-SlowHTTPTest': 'DoS: SlowHTTPTest',
    'DoS attacks-Hulk': 'DoS: Hulk',
    'DDOS attack-LOIC-HTTP': 'DDoS: LOIC-HTTP',
    'DDOS attack-HOIC': 'DDoS: HOIC',
    'Brute Force -Web': 'Brute Force: Web',
    'Brute Force -XSS': 'Brute Force: XSS',
    'SQL Injection': 'Web Attack: SQL Injection',
    'FTP-BruteForce': 'Brute Force: FTP',
    'SSH-Bruteforce': 'Brute Force: SSH',
    'Infilteration': 'Infiltration',
    'Bot': 'Botnet'
}

# Dashboard configuration
DASHBOARD_PORT = 8501
DASHBOARD_HOST = 'localhost'

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
