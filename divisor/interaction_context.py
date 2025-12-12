from dataclasses import dataclass
from typing import Callable

import torch

from nnll.random import RNGState


@dataclass
class InteractionContext:
    """Container for functions and utilities needed during interactive routing."""

    clear_prediction_cache: Callable[[], None]
    rng: RNGState
    variation_rng: RNGState
    ae: torch.nn.Module | None = None
    t5: torch.nn.Module | None = None
    clip: torch.nn.Module | None = None
    recompute_text_embeddings: Callable[[str], None] | None = None
