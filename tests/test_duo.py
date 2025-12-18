# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from divisor.duo import configuration_duo, modeling_duo
from transformers import AutoModelForMaskedLM, AutoTokenizer
from nnll.init_gpu import device

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained("exdysa/duo").to(device)
prompt = tokenizer.encode("Hello, I'm a language model", return_tensors="pt")
gen_out = model.forward(prompt)

tokenizer.decode(gen_out[0])
