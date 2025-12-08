from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate

model_name = "Gen-Verse/TraDo-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "What's the solution of x^2 - 2x + 1 = 0\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokens = tokenizer.batch_encode_plus([text], return_tensors="pt", padding=True, truncation=True, max_length=200)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

output_ids = block_diffusion_generate(
    model,
    prompt=tokens,
    mask_id=151669,
    gen_length=200,
    block_length=4,
    denoising_steps=4,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy="low_confidence_dynamic",
    confidence_threshold=0.9,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
cleaned_text = output_text.replace("<|MASK|>", "").replace("<|endoftext|>", "")
print(cleaned_text)
