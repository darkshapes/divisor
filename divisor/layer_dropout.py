# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from typing import Callable, Any
from nnll.console import nfo
from torch.nn import ModuleList


def process_blocks_with_dropout(
    blocks: ModuleList,
    layer_dropouts: list[int] | None,
    start_index: int,
    block_type: str,
    process_block: Callable[[Any, Any], Any],
    state: Any,
) -> Any:
    """Process blocks, skipping layer dropout blocks\n
    :param blocks: List of blocks to process
    :param layer_dropouts: List of block indices to skip (None to skip none)
    :param start_index: Starting index for layer dropout checking (for offset)
    :param block_type: Type of block ("double" or "single") for logging
    :param process_block: Callable that takes (block, current_state) and returns updated state
    :param state: Initial state to process (e.g., (img, txt) for double, img for single)
    :return: The final state after processing all non-dropped blocks
    """
    for block_index, block in enumerate(blocks):
        global_index = start_index + block_index
        if layer_dropouts is not None and global_index in layer_dropouts:
            nfo(f"Dropping layer {global_index} ({block_type} block)")
            continue

        state = process_block(block, state)

    return state
