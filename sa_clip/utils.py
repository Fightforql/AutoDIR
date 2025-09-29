from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_checkpoint(state: Dict[str, Any], output_dir: str, step: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{step:07d}.pt")
    torch.save(state, path)
    return path