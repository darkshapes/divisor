# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

from nnll.console import nfo
from divisor.registry import gfx_device, gfx_dtype
import torch
import torch.nn.functional as F


from divisor.flux1.loading import load_mmada_model
from divisor.mmada.live_token import (
    get_highlighted_text_tuples,
    get_num_transfer_tokens,
)
from divisor.noise import add_gumbel_noise
from divisor.mmada.sampling import prepare
from divisor.mmada.system_messages import THINKING_MODE_LM_PROMPT
from divisor.mmada.text_embedder import HFEmbedder
from divisor.spec import (
    InitialParamsMMaDA,
    MMaDAParams,
    ModelSpec,
    get_model_spec,
    mmada_configs,
)


def clear_outputs_action():
    return None, None


@torch.no_grad()
def generate_viz_wrapper_lm(
    mir_id,
    prompt_text,
    steps,
    gen_length,
    block_length,
    temperature,
    cfg_scale,
    remasking_strategy,
    thinking_mode_lm,
):
    precision = gfx_dtype
    model_spec: ModelSpec = get_model_spec(mir_id, mmada_configs)
    if not isinstance(model_spec.params, MMaDAParams) or not isinstance(model_spec.init, InitialParamsMMaDA):
        raise TypeError(
            f"MMaDA spec not found for: {mir_id} \
                with params type {type(model_spec.params).__name__} \
                    and init type {type(model_spec.init).__name__}",
        )
    else:
        mask_id = model_spec.init.mask_id
        max_position_embeddings = model_spec.init.max_position_embeddings
        model = load_mmada_model(model_spec, device=gfx_device)
        hf = HFEmbedder(model_spec.repo_id, max_length=max_position_embeddings)
    if thinking_mode_lm:
        prompt_text = THINKING_MODE_LM_PROMPT + prompt_text

    input_ids = prepare(model, hf.tokenizer, prompt_text)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    raw_prompt_attention_mask = None

    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=gfx_device)
    x[:, :prompt_len] = input_ids.clone()
    nfo(f"Starting generation: Prompt ({prompt_len} tokens) + Initial Masks")
    # yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), "Starting generation: Prompt + Initial Masks"
    yield (
        get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask),
        f"Starting generation: Prompt ({prompt_len} tokens) + Initial Masks",
    )

    if gen_length == 0:
        final_text_output = hf.tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
        yield get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask), final_text_output[0] if final_text_output else ""
        return

    if block_length <= 0 or gen_length % block_length != 0:
        yield (
            get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask),
            f"Error: gen_length ({gen_length}) must be divisible by block_length ({block_length}) and block_length > 0.",
        )
        return
    num_blocks = gen_length // block_length

    if steps <= 0 or steps % num_blocks != 0:
        yield (
            get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask),
            f"Error: steps ({steps}) must be positive and divisible by num_blocks ({num_blocks}). Steps: {steps}, Num Blocks: {num_blocks}",
        )
        return
    steps_per_block = steps // num_blocks

    for num_block_iter in range(num_blocks):
        current_block_start_idx_in_x = prompt_len + num_block_iter * block_length
        current_block_end_idx_in_x = prompt_len + (num_block_iter + 1) * block_length

        block_masks_bool_current = torch.zeros_like(x, dtype=torch.bool)
        block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x] = x[:, current_block_start_idx_in_x:current_block_end_idx_in_x] == mask_id

        num_transfer_tokens_for_this_block = get_num_transfer_tokens(block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x], steps_per_block)

        for i_step_in_block in range(steps_per_block):
            mask_index_global = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                # For unconditional pass, mask out the original prompt tokens that are not padding
                # raw_prompt_attention_mask is (B, prompt_len)
                if raw_prompt_attention_mask is not None:
                    prompt_active_tokens_mask = raw_prompt_attention_mask.bool()  # True where actual prompt tokens are
                    un_x[:, :prompt_len][prompt_active_tokens_mask] = mask_id
                else:
                    # If no attention mask, mask all prompt tokens
                    un_x[:, :prompt_len] = mask_id

                x_cfg_input = torch.cat([x, un_x], dim=0)
                # Pass attention_mask for CFG if model expects it, covering both parts
                # For simplicity, not passing explicit attention_mask here; relies on model's internal handling.
                model_output = model(x_cfg_input)
                logits_cond, logits_uncond = torch.chunk(model_output.logits, 2, dim=0)
                logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
            else:
                # Not passing explicit attention_mask here; relies on model's internal handling.
                model_output = model(x)
                logits = model_output.logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0_predicted_tokens = torch.argmax(logits_with_noise, dim=-1)

            if remasking_strategy == "low_confidence":
                probs = F.softmax(logits.to(precision), dim=-1)
                x0_probs = torch.gather(probs, dim=-1, index=x0_predicted_tokens.unsqueeze(-1)).squeeze(-1)
            elif remasking_strategy == "random":
                x0_probs = torch.rand(x.shape, device=x.device, dtype=precision)
            else:
                yield (
                    get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask),
                    f"Error: Unknown remasking strategy '{remasking_strategy}'",
                )
                return

            confidence_for_selection = torch.full_like(x0_probs, -torch.inf)
            candidate_positions_for_unmasking = mask_index_global & block_masks_bool_current
            confidence_for_selection = torch.where(candidate_positions_for_unmasking, x0_probs, -torch.inf)

            x0_final_candidates = torch.where(mask_index_global, x0_predicted_tokens, x)

            transfer_indices_bool = torch.zeros_like(x, dtype=torch.bool)
            num_to_transfer_this_step_batch = num_transfer_tokens_for_this_block[:, i_step_in_block]

            for j_batch_idx in range(batch_size):
                k_val = min(num_to_transfer_this_step_batch[j_batch_idx].item(), candidate_positions_for_unmasking[j_batch_idx].sum().item())  # ensure k isn't too large

                if k_val > 0:
                    # Ensure confidence_for_selection[j_batch_idx] is 1D for topk
                    conf_slice = confidence_for_selection[j_batch_idx]
                    if conf_slice.ndim > 1:
                        conf_slice = conf_slice.view(-1)  # Should already be 1D from x0_probs

                    # Check if there are enough valid (non -inf) confidences
                    valid_conf_count = (conf_slice > -torch.inf).sum().item()
                    actual_k = min(k_val, valid_conf_count)

                    if actual_k > 0:
                        _, topk_indices_in_x = torch.topk(conf_slice, k=int(actual_k))
                        transfer_indices_bool[j_batch_idx, topk_indices_in_x] = True

            x[transfer_indices_bool] = x0_final_candidates[transfer_indices_bool]

            current_total_step = num_block_iter * steps_per_block + i_step_in_block + 1
            total_overall_steps = num_blocks * steps_per_block
            status_msg = f"Block {num_block_iter + 1}/{num_blocks}, Step {i_step_in_block + 1}/{steps_per_block} (Total: {current_total_step}/{total_overall_steps})"
            yield get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask), status_msg

    final_generated_ids = x[:, prompt_len:]
    final_text_output = hf.tokenizer.batch_decode(final_generated_ids, skip_special_tokens=True)

    final_text_str = final_text_output[0] if final_text_output and len(final_text_output) > 0 else ""
    yield get_highlighted_text_tuples(x, input_ids, prompt_len, hf.tokenizer, mask_id, raw_prompt_attention_mask), final_text_str
