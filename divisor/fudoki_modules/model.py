from transformers import AutoModelForCausalLM
from divisor.fudoki_modules.janus.models.modeling_vlm import MultiModalityCausalLM


def instantiate_model(pretrained_weight_path):
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(pretrained_weight_path, trust_remote_code=True)  # type: ignore
    model = vl_gpt
    return model
