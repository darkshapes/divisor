from dataclasses import dataclass
from typing import Callable, Optional

from divisor.state import RNGState


@dataclass
class InteractionContext:
    """Container for functions and utilities needed during interactive routing."""

    clear_prediction_cache: Callable[[], None]
    rng: RNGState | None = None
    variation_rng: RNGState | None = None
    ae: Optional[torch.nn.Module] | None = None
    t5: Optional[torch.nn.Module] | None = None
    clip: Optional[torch.nn.Module] | None = None
    recompute_text_embeddings: Optional[Callable[[str], None]] | None = None
