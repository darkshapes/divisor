# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

import torch

from divisor.mmada.system_messages import THINKING_MODE_LM_PROMPT


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """

    mask_num = mask_index.sum(dim=1, keepdim=True)
    # Ensure steps is at least 1 to avoid division by zero if mask_num is also 0 (though sum should be >=0)
    steps = max(1, int(steps))  # Ensure steps is a positive integer
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    for i in range(mask_num.size(0)):  # Iterate over batch
        if remainder[i] > 0:  # Ensure remainder is positive before indexing
            num_transfer_tokens[i, : remainder[i].item()] += 1  # .item() for single value tensor to int
    return num_transfer_tokens


def get_highlighted_text_tuples(current_x_ids_batch, prompt_input_ids, prompt_len, tk, current_mask_id, raw_prompt_attention_mask):
    if current_x_ids_batch is None or current_x_ids_batch.ndim == 0 or current_x_ids_batch.shape[0] == 0:
        return [("Error in sequence data for visualization.", "ERROR")]
    # only answer part
    current_x_ids_batch = current_x_ids_batch[:, prompt_len:]
    seq_ids = current_x_ids_batch[0].tolist()
    eos_token_id = tk.eos_token_id  # Get EOS token ID

    # Stage 1: Build initial list of tuples with (token_str, label, token_id_int)
    # This helps in identifying EOS tokens later without re-checking the type.
    intermediate_tuples = []
    for j, token_id_int in enumerate(seq_ids):
        try:
            token_str = tk.decode([token_id_int], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception:  # Handle cases where a token ID might be problematic (e.g. with mock)
            token_str = f"[ID:{token_id_int}]"

        label = "ERROR"
        if token_id_int == current_mask_id:
            token_str = "[MASK]"
            label = "MASK"
        else:
            label = "GEN"
        intermediate_tuples.append((token_str, label, token_id_int))

    return intermediate_tuples


def get_input_ids(prompt_text, thinking_mode_lm, tokenizer, model_id, device):
    if thinking_mode_lm:
        prompt_text = THINKING_MODE_LM_PROMPT + prompt_text

    try:
        m = [{"role": "user", "content": prompt_text}]
        processed_prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        yield [("Error applying chat template.", "ERROR")], f"Chat template error: {e}"
        processed_prompt_text = prompt_text
    try:
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:  # Should have been caught by load_model, but double check
                yield [("Tokenizer Error", "ERROR")], "pad_token_id is not set in tokenizer."
                return

        input_ids = tokenizer(
            text=processed_prompt_text,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            truncation=True,
            max_length=model_id.config.max_position_embeddings if hasattr(model_id.config, "max_position_embeddings") else 2048,
        )["input_ids"].to(model_id)
        raw_prompt_attention_mask = None

    except Exception as e:
        yield [("Error tokenizing prompt.", "ERROR")], f"Tokenization error: {e}"
        return

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    return batch_size, prompt_len, input_ids, raw_prompt_attention_mask
