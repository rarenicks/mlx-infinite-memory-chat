import psutil
import mlx_lm
from mlx_lm import load, generate
import os

# Constants
MIN_MEMORY_GB = 8  # Minimum RAM required to attempt loading 8B 4-bit safely
MODEL_PATH_DEFAULT = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

def check_memory():
    """
    Checks available system memory.
    Returns (bool, str): (is_safe, message)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    
    print(f"System Memory: Total={total_gb:.1f}GB, Available={available_gb:.1f}GB")
    
    if total_gb < MIN_MEMORY_GB:
        return False, f"Total system memory ({total_gb:.1f}GB) is below the recommended {MIN_MEMORY_GB}GB for this model."
    
    # Warning if available memory is low, but don't hard block if swap is available (macOS handles this well)
    if available_gb < 4:
        print(f"WARNING: Available memory is low ({available_gb:.1f}GB). Performance may degrade.")
        
    return True, "Memory check passed."

def load_model(model_path=MODEL_PATH_DEFAULT):
    """
    Loads the model and tokenizer.
    """
    print(f"Loading model from {model_path}...")
    
    # Check if local path exists, otherwise treat as HF repo ID
    if os.path.exists("models/Llama-3.1-8B-Instruct-4bit"):
        model_path = "models/Llama-3.1-8B-Instruct-4bit"
        print(f"Found local model at {model_path}")
        
    try:
        model, tokenizer = load(model_path)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
