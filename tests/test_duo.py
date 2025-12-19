# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from nnll.init_gpu import device
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nnll.random import RNGState

from divisor.contents import get_dtype
from divisor.duo import configuration_duo, modeling_duo

rnd = RNGState()

precision = get_dtype(device)


@torch.inference_mode()
def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForMaskedLM.from_pretrained("exdysa/duo", dtype=precision)
    prompt = """What is the capital of France?"""
    print(f"prompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    model.to(device)
    output = model(input_ids)

    # Handle both return_dict and tuple returns
    if hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output[0] if isinstance(output, tuple) else output
    token_ids = logits.argmax(dim=-1)[0]  # [0] selects first batch element
    decoded = tokenizer.decode(token_ids)
    print(f"decoded response: {decoded}")


if __name__ == "__main__":
    main()
