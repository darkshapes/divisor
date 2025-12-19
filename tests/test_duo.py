# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from nnll.init_gpu import device
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nnll.random import RNGState

from divisor.contents import get_dtype
from divisor.duo import configuration_duo, modeling_duo

rnd = RNGState()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained("exdysa/duo")
prompt = """
: boundaries are good to set
[09:06]clyde
: whenever i say this its more like "i have no preference do u have anything in mind you prefer to do"
[09:06]: its good self care
[09:06]: yeah i know but i also feel like
[09:07]: maybe its not that you dont have a pref, maybe its that you dont know your pref?
[09:07]: like at some point everyone has a pref
[09:07]clyde
: im rlly kinda chill with everything
[09:07]: its a bit about finding the deep you in things
[09:07]: yes me too
[09:07]: but i also have practiced being more immediate in speaking my needs
[09:09]clyde
: i guess i never thought of it that way

"""
print(f"prompt: {prompt}")
input_ids = tokenizer.encode(prompt, return_tensors="pt")
model.to(device).eval()
generator = torch.Generator(device=device).manual_seed(rnd.next_seed())
output = model(
    input_ids,
    # max_new_tokens=100,
)
# Handle both return_dict and tuple returns
if hasattr(output, "logits"):
    logits = output.logits
else:
    logits = output[0] if isinstance(output, tuple) else output
token_ids = logits.argmax(dim=-1)[0]  # [0] selects first batch element
decoded = tokenizer.decode(token_ids)
print(f"decoded response: {decoded}")
