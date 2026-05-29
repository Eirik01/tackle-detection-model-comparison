"""
Central configuration for the tackle-detection-model-comparison project.
All paths and settings should be defined here.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT /  "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# REMOTE paths (if applicable) USED FOR PROCESSING AND STORING LARGE DATASETS on FOX HPC and ONEDRIVE
ONEDRIVE_FOLDER_NAME = "OneDrive - Universitetet i Oslo"

# FOX Project work Area for temporary storage (large, regenerable data:
# features, predictions, saved models, results). Override via the
# FOX_DATADIR_PATH env var when running off-cluster; defaults to the UiO FOX
# HPC work area.
FOX_DATADIR_PATH = Path(os.getenv("FOX_DATADIR_PATH", "/cluster/work/projects/ec12/ec-eirikto"))
FOX_SOCCERNET_DATADIR = FOX_DATADIR_PATH / "soccernet"
FOX_PREDICTIONS_DATADIR = FOX_DATADIR_PATH / "predictions"
FOX_MODELS_DATADIR = FOX_DATADIR_PATH / "saved_models"

# FOX Project Area for permanent storage (TACDEC videos + labels). Override
# via the FOX_PROJECT_AREA_PATH env var when running off-cluster; defaults to
# the UiO FOX HPC project area.
FOX_PROJECT_AREA_PATH = Path(os.getenv("FOX_PROJECT_AREA_PATH", "/fp/projects01/ec12/ec-eirikto"))
TACDEC_VIDEOS = FOX_PROJECT_AREA_PATH / "TACDEC" / "videos"
TACDEC_LABELS = FOX_PROJECT_AREA_PATH / "TACDEC" / "labels"
TACDEC_MODELS = FOX_DATADIR_PATH / "TACDEC" / "models"
TACDEC_CROPS = FOX_PROJECT_AREA_PATH / "TACDEC" / "crops"

# Features live on the work area: large (~135 GB at 25 fps with dense),
# regenerable, and the project area is essentially full. Subject to the
# work-directory cleanup policy.
TACDEC_FEATURES = FOX_DATADIR_PATH / "TACDEC" / "features"
TACDEC_FEATURES_DINOV3 = TACDEC_FEATURES / "dinov3_large"
TACDEC_FEATURES_VJEPA2 = TACDEC_FEATURES / "vjepa2_large"
TACDEC_RESULTS = FOX_DATADIR_PATH / "TACDEC" / "results"

# Work-area root for the untrimmed_footage_experiment/ qualitative run. The experiment
# scripts own what they download / extract / predict; this is just where it
# lands.
SOCCERNET_EXPERIMENT_DIR = FOX_DATADIR_PATH / "soccernet_thesis_experiment"


# ============================================================================
# BACKBONE CONFIGURATION
# ============================================================================
# Switch between different visual encoders for comparison experiments
# Options: 'dinov3' or 'vjepa2'
BACKBONE_TYPE = os.getenv("BACKBONE_TYPE", "dinov3")

# Model size: 'base', 'large', 'huge', or 'giant'
BACKBONE_SIZE = os.getenv("BACKBONE_SIZE", "large")

# Backbone-specific configurations
BACKBONE_CONFIGS = {
    'dinov3': {
        'base': {
            'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'feature_dim': 768,
            'requires_token': True
        },
        'large': {
            'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'feature_dim': 1024,
            'requires_token': True
        }
    },
    'vjepa2': {
        # NOTE: V-JEPA2 does not have a "base" model - smallest is "large"
        # All V-JEPA2 models output 1024-dim features (encoder hidden_size)
        'large': {
            'model_name': 'facebook/vjepa2-vitl-fpc64-256',
            'feature_dim': 1024,
            'requires_token': False
        },
        'huge': {
            'model_name': 'facebook/vjepa2-vith-fpc64-256',
            'feature_dim': 1024,  # Same as Large
            'requires_token': False
        },
        'giant': {
            'model_name': 'facebook/vjepa2-vitg-fpc64-256',
            'feature_dim': 1024,  # Same as Large
            'requires_token': False
        }
    }
}

# Get current backbone configuration
def get_backbone_config():
    """
    Returns the configuration for the currently selected backbone.
    
    Returns:
        dict: Configuration with 'model_name', 'feature_dim', 'requires_token'
    """
    if BACKBONE_TYPE not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown BACKBONE_TYPE: {BACKBONE_TYPE}. Must be 'dinov3' or 'vjepa2'")
    
    if BACKBONE_SIZE not in BACKBONE_CONFIGS[BACKBONE_TYPE]:
        raise ValueError(f"Unknown BACKBONE_SIZE: {BACKBONE_SIZE}. Must be 'base' or 'large'")
    
    return BACKBONE_CONFIGS[BACKBONE_TYPE][BACKBONE_SIZE]

# Dynamic configuration based on selected backbone
_config = get_backbone_config()
MODEL_NAME = _config['model_name']
FEATURE_DIM = _config['feature_dim']
REQUIRES_HF_TOKEN = _config['requires_token']

# Model identifier for file naming (e.g., 'dinov3_b', 'vjepa2_l')
BACKBONE_ID = f"{BACKBONE_TYPE}_{BACKBONE_SIZE[0]}"

# Hugging Face token (only needed for some models)
HF_TOKEN = os.getenv("DINOv3_key") if REQUIRES_HF_TOKEN else None

# SoccerNet download configuration
SOCCERNETV2_PASSWORD = os.getenv("SoccerNetv2_password")

# Default extraction parameters
DEFAULT_BATCH_SIZE = 32

# ============================================================================
# CLASSIFICATION CONFIGURATION
# ============================================================================
# Number of output classes for action spotting
# Options:
#   5: [Tackle-Live, Tackle-Replay, Live-Incomplete, Replay-Incomplete, Background]
#   3: [Tackle-Live, Tackle-Replay, Background] (merges incomplete into parent classes)
NUM_CLASSES = 3  # Change to 5 to use full class set

# Device configuration
DEVICE = "cuda" if os.getenv("FORCE_CPU", "").lower() != "true" else "cpu"

# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================
# Random seed for training, validation, and data splits
# Set via SLURM: export TRAIN_SEED=42 (or any integer)
TRAIN_SEED = int(os.getenv("TRAIN_SEED", "42"))

# Random seed for feature extraction (for scientific rigor)
# Set via SLURM: export EXTRACTION_SEED=42 (or any integer)
EXTRACTION_SEED = int(os.getenv("EXTRACTION_SEED", "42"))

# Print current configuration
def print_config():
    """Print current backbone configuration."""
    print("="*60)
    print("Current Configuration")
    print("="*60)
    print(f"Backbone: {BACKBONE_TYPE.upper()} ({BACKBONE_SIZE})")
    print(f"Model: {MODEL_NAME}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Backbone ID: {BACKBONE_ID}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Device: {DEVICE}")
    print("="*60)
