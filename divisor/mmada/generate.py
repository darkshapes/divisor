# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from nnll.init_gpu import device
from divisor.mmada.modeling_mmada import MMadaModelLM, add_gumbel_noise, get_num_transfer_tokens

if device.type == "mps":
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float64


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=126336, attention_mask=None):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        print(f"attention_bias: {attention_bias}")
    else:
        attention_bias = None
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length :] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch_dtype), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            # print(confidence.shape)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Early stop if no masks remain
            if not mask_index.any():
                break

    return x


def main(model_id: str):
    from nnll.init_gpu import device

    model = MMadaModelLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32).to(device.type).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text=prompt, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
    input_ids = input_ids.to(device)
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=128, temperature=0.0, cfg_scale=3.0, remasking="low_confidence")
    print(tokenizer.batch_decode(out, skip_special_tokens=True))
    # Extract generated tokens (everything after the input)
    generated_tokens = out[:, input_ids.shape[1] :]
    print(generated_tokens)

    # Debug: check what tokens were generated
    print(f"Generated token shape: {generated_tokens.shape}")
    print(f"Generated token IDs: {generated_tokens[0, :20].tolist()}")  # First 20 tokens
    print(f"Mask token count: {(generated_tokens == 126336).sum().item()}")

    # Filter out mask tokens before decoding
    mask_id = 126336
    # Replace mask tokens with a padding token or remove them

    # Find where generation actually ends (first mask token, end token 126081, or padding)
    mask_id = 126336
    end_token_id = 126081
    batch_size = generated_tokens.shape[0]
    actual_lengths = []

    for b in range(batch_size):
        # Find first mask token or end token position
        mask_positions = (generated_tokens[b] == mask_id).nonzero(as_tuple=True)[0]
        end_positions = (generated_tokens[b] == end_token_id).nonzero(as_tuple=True)[0]

        positions = []
        if len(mask_positions) > 0:
            positions.append(mask_positions[0].item())
        if len(end_positions) > 0:
            positions.append(end_positions[0].item())

        if positions:
            actual_length = min(positions)
        else:
            actual_length = generated_tokens.shape[1]
        actual_lengths.append(actual_length)

    # Trim to actual generated content (before first mask or end token)
    max_length = max(actual_lengths) if actual_lengths else generated_tokens.shape[1]
    generated_tokens_trimmed = generated_tokens[:, :max_length]

    decoded_trimmed_true = tokenizer.batch_decode(generated_tokens_trimmed, skip_special_tokens=True)
    print(f"Decoded trimmed (with special tokens): {decoded_trimmed_true}")
    decoded_trimmed_false = tokenizer.batch_decode(generated_tokens_trimmed, skip_special_tokens=False)
    print(f"Decoded trimmed (without special tokens): {decoded_trimmed_false}")

    # For now, let's decode and see what we get
    decoded_true = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"Decoded output (with special tokens): {decoded_true}")
    decoded_false = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    print(f"Decoded output (without special tokens): {decoded_false[0]}")  # First 200 chars


if __name__ == "__main__":
    main(model_id="Gen-Verse/MMaDA-8B-MixCoT")
