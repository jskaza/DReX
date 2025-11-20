import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_hf_token():
    """Load Hugging Face token from file."""
    try:
        return open("hf_token.txt").read().strip()
    except FileNotFoundError:
        return None
