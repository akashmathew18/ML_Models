"""Configuration settings for the ML Models application."""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model configurations
MODEL_CONFIGS = {
    "kNN": {
        "n_neighbors": 5,
        "model_path": os.path.join(MODEL_DIR, "KNN_Model.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "KNN_Scaler.pkl")
    },
    "LR": {
        "model_path": os.path.join(MODEL_DIR, "LR_Model.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "LR_Scaler.pkl")
    },
    "PR": {
        "degree": 2,
        "model_path": os.path.join(MODEL_DIR, "Poly_Model.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "Poly_Scaler.pkl"),
        "transformer_path": os.path.join(MODEL_DIR, "Poly_Transformer.pkl")
    },
    "SLR": {
        "model_path": os.path.join(MODEL_DIR, "SLR_Model.pkl")
    },
    "MLR": {
        "model_path": os.path.join(MODEL_DIR, "MLR_Model.pkl")
    }
}

# Flask configurations
FLASK_CONFIG = {
    "DEBUG": True,
    "HOST": "0.0.0.0",
    "PORT": 5000
}

# Logging configuration
LOG_CONFIG = {
    "filename": os.path.join(BASE_DIR, "logs", "app.log"),
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
