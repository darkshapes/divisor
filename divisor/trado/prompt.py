# adapted from SADR https://github.com/JetAstra/SDAR/blob/main/README.md and https://huggingface.co/JetAstra/SDAR/README.md


from nnll.init_gpu import device
from divisor.trado.spec import configs, ModelSpec


def generate_cuda(model_spec: ModelSpec):
    import os
    from transformers import AutoTokenizer
    from jetengine import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_spec.repo_id, trust_remote_code=True)
    # Initialize the LLM
    llm = LLM(
        model_spec.repo_id,
        enforce_eager=True,
        tensor_parallel_size=1,
        mask_token_id=151669,  # Optional: only needed for masked/diffusion models
        block_length=4,
    )

    # Set sampling/generation parameters
    sampling_params = SamplingParams(
        temperature=1.0, topk=0, topp=1.0, max_tokens=256, remasking_strategy="low_confidence_dynamic", block_length=4, denoising_steps=4, dynamic_threshold=0.9
    )

    # Prepare a simple chat-style prompt
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Explain what reinforcement learning is in simple terms."}], tokenize=False, add_generation_prompt=True)

    # Generate text
    outputs = llm.generate_streaming([prompt], sampling_params)


def generate(model_spec: ModelSpec):
    from divisor.trado.modeling_sdar import SDARForCausalLM
    from divisor.trado.tokenization_qwen2_fast import Qwen2TokenizerFast
    from divisor.trado.generate import block_diffusion_generate

    model = SDARForCausalLM.from_pretrained(model_spec.repo_id, trust_remote_code=True, torch_dtype="bfloat16")
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_spec.repo_id, trust_remote_code=True)

    prompt = "Word\n"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokens = tokenizer._encode_plus([text], return_tensors="pt", padding=True, truncation=True, max_length=200)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        params=model_spec.params,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace("<|MASK|>", "").replace("<|endoftext|>", "")
    print(cleaned_text)


def main(model_id: str):
    model_spec = configs[model_id]["*"]
    if device.type == "cuda":
        try:
            import jetengine

            generate_cuda(model_spec)
        except (ModuleNotFoundError, ImportError, NameError):
            generate(model_spec)
    else:
        generate(model_spec)
