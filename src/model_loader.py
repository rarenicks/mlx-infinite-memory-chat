import psutil
import mlx_lm
from mlx_lm import load, generate
import os
from src.logging_config import setup_logger

logger = setup_logger(__name__)

# Constants
# Constants
MIN_MEMORY_GB = 8  # Minimum RAM required to attempt loading 8B 4-bit safely
MODEL_PATH_DEFAULT = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

def list_available_models():
    """
    Lists all subdirectories in the 'models' directory.
    Returns a list of model names.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    return sorted(models)

def check_memory():
    """
    Checks available system memory.
    Returns (bool, str): (is_safe, message)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    
    logger.info(f"System Memory: Total={total_gb:.1f}GB, Available={available_gb:.1f}GB")
    
    if total_gb < MIN_MEMORY_GB:
        return False, f"Total system memory ({total_gb:.1f}GB) is below the recommended {MIN_MEMORY_GB}GB for this model."
    
    # Warning if available memory is low, but don't hard block if swap is available (macOS handles this well)
    if available_gb < 4:
        logger.warning(f"Available memory is low ({available_gb:.1f}GB). Performance may degrade.")
        
    return True, "Memory check passed."

def load_model(model_path=MODEL_PATH_DEFAULT):
    """
    Loads the model and tokenizer.
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Check if local path exists, otherwise treat as HF repo ID
    if os.path.exists("models/Llama-3.1-8B-Instruct-4bit"):
        model_path = "models/Llama-3.1-8B-Instruct-4bit"
        logger.info(f"Found local model at {model_path}")
        
    try:
        model, tokenizer = load(model_path)
        logger.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e
