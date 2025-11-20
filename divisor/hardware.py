import torch
import gc
from typing import Literal, Optional
from numpy import random

torch.backends.cudnn.deterministic = False

torch.mps.set_rng_state
torch.cuda.set_rng_state


def set_torch_device(
    device_override: Optional[Literal["cuda", "mps", "cpu"]] = None,
) -> torch.device:
    """Set the PyTorch device, with optional manual override.\n
    :param device_override: Optional device to use. "cuda", "mps", or "cpu"
    :returns: The selected torchdevice
    :raises ValueError: If device_override is not one of the allowed values"""
    if device_override is not None:
        if device_override not in ("cuda", "mps", "cpu"):
            raise ValueError(f"device_override must be one of 'cuda', 'mps', or 'cpu', got '{device_override}'")
        return torch.device(device_override)

    # Auto-detect best available device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    return device


device = set_torch_device()
dtype = torch.float16
seed = torch.random.seed()


def seed_planter(seed: int = torch.random.seed()) -> int:
    """Force seed number to all available devices\n
    :param seed: The number to grow all random generation from, defaults to `soft_random` function
    :return: The `int` seed that was provided to the functions."""

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    if "cuda" in device.type:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if "mps" in device.type:
        torch.mps.manual_seed(seed)
    return seed


def clear_cache(device_override: Optional[Literal["cuda", "mps", "cpu"]] = None):
    gc.collect()
    if device.type == "cuda" or device_override == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" or device_override == "mps":
        torch.mps.empty_cache()
