"""Small cross-cutting utilities (reproducibility)."""

from __future__ import annotations

import os
import random

import numpy as np

from . import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
