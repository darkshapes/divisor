# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

from dataclasses import dataclass
from pathlib import Path


import torch
from huggingface_hub import snapshot_download
from nnll.init_gpu import device
from safetensors.torch import load_file as load_sft
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from nnll.console import nfo

from divisor.flux_modules.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux_modules.model import Flux, FluxLoraWrapper, FluxParams
from divisor.flux_modules.text_embedder import HFEmbedder


def retrieve_model(repo_id: str, file_name: str) -> Path:
    """Get the local path for a checkpoint file, downloading if necessary.
    :param repo_id: Repository ID for the checkpoint
    :param file_name: Name of the checkpoint file
    :returns: Path to the checkpoint file
    """

    model_dir = snapshot_download(repo_id=repo_id, allow_patterns=[file_name])
    return Path(model_dir) / file_name


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        nfo(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        nfo("\n" + "-" * 79 + "\n")
        nfo(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        nfo(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        nfo(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


@dataclass
class CompatibilitySpec:
    repo_id: str
    file_name: str


@dataclass
class InitialParams:
    num_steps: int
    max_length: int
    guidance: float
    shift: bool
    width: int = 1360
    height: int = 768


@dataclass
class ModelSpec:
    repo_id: str
    file_name: str
    params: FluxParams | AutoEncoderParams | type[AutoencoderTiny] | type[FluxLoraWrapper]
    init: InitialParams | None = None


configs = {
    "model.dit.flux1-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            file_name="flux1-dev.safetensors",
            init=InitialParams(
                num_steps=28,
                max_length=512,
                guidance=4.0,
                shift=True,
            ),
            params=FluxParams(
                in_channels=64,
                out_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=True,
            ),
        ),
        "fp8-sai": CompatibilitySpec(
            repo_id="Comfy-Org/flux1-dev",
            file_name="flux1-dev-fp8.safetensors",
        ),
    },
    "model.vae.flux1-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            file_name="ae.safetensors",
            params=AutoEncoderParams(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            ),
        ),
    },
    "model.taesd.flux1-dev": {
        "*": ModelSpec(repo_id="madebyollin/taef1", file_name="diffusion_pytorch_model.safetensors", params=AutoencoderTiny),
    },
    "model.dit.flux1-schnell": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-schnell",
            file_name="flux1-schnell.safetensors",
            init=InitialParams(
                num_steps=4,
                max_length=256,
                guidance=2.5,
                shift=False,
            ),
            params=FluxParams(
                in_channels=64,
                out_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=False,
            ),
        ),
        "fp8-sai": CompatibilitySpec(
            repo_id="Comfy-Org/flux1-dev",
            file_name="flux1-dev-fp8.safetensors",
        ),
    },
}


def get_model_spec(mir_id: str) -> ModelSpec:
    """Get the base ModelSpec for a given model ID.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :returns: The base ModelSpec from the "*" key
    """
    if mir_id not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model ID: {mir_id}. Available: {available}")

    config_dict = configs[mir_id]
    if "*" not in config_dict:
        raise ValueError(f"Model {mir_id} does not have a base spec (missing '*' key)")

    base_spec = config_dict["*"]
    if not isinstance(base_spec, ModelSpec):
        raise ValueError(f"Model {mir_id} base spec is not a ModelSpec")

    return base_spec


def get_compatibility_spec(mir_id: str, compatibility_key: str) -> CompatibilitySpec | None:
    """Get a compatibility spec for a model if available.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param compatibility_key: Compatibility key (e.g., "fp8-sai")
    :returns: CompatibilitySpec if available, None otherwise
    """
    if mir_id not in configs:
        return None

    config_dict = configs[mir_id]
    compat_spec = config_dict.get(compatibility_key)

    if compat_spec is None:
        return None

    if not isinstance(compat_spec, CompatibilitySpec):
        raise ValueError(f"Compatibility spec {compatibility_key} for {mir_id} is not a CompatibilitySpec")

    return compat_spec


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """Optionally expand the state dict to match the model's parameters shapes.\n
    :param model: The model to match parameters against
    :param state_dict: The state dictionary to expand
    :returns: The expanded state dictionary
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                nfo(f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}.")
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict


def load_state_dict_into_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a state dict into a model with optional expansion.\n
    :param model: The model to load weights into
    :param state_dict: The state dictionary to load
    :param verbose: Whether to print warnings
    :returns: Tuple of (missing_keys, unexpected_keys)
    """
    expanded_sd = optionally_expand_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(expanded_sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)
    return missing, unexpected


def load_lora_weights(
    model: FluxLoraWrapper,
    lora_repo_id: str,
    lora_filename: str,
    device: str | torch.device = device,
    verbose: bool = True,
) -> None:
    """Load LoRA weights into a FluxLoraWrapper model.\n
    :param model: The FluxLoraWrapper model
    :param lora_repo_id: Repository ID for the LoRA checkpoint
    :param lora_filename: Filename of the LoRA checkpoint
    :param device: Device to load weights on
    :param verbose: Whether to print warnings
    """
    nfo("Loading LoRA")
    lora_path = str(retrieve_model(lora_repo_id, lora_filename))
    lora_sd = load_sft(lora_path, device=str(device))
    # loading the lora params + overwriting scale values in the norms
    load_state_dict_into_model(model, lora_sd, verbose=verbose)


def load_flow_model(
    mir_id: str,
    device: str | torch.device = device,
    repo_id: str | None = None,
    file_name: str | None = None,
    verbose: bool = True,
    lora_repo_id: str | None = None,
    lora_filename: str | None = None,
) -> Flux:
    """Load a flow model (DiT model).\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param device: Device to load the model on
    :param verbose: Whether to print loading warnings
    :param repo_id: Optional repository ID to override the base spec's repo_id
    :param file_name: Optional file name to override the base spec's file_name
    :param lora_repo_id: Optional LoRA repository ID (if not in config)
    :param lora_filename: Optional LoRA filename (if not in config)
    :returns: Loaded Flux model"""

    nfo("Init model")
    config = get_model_spec(mir_id)

    if not isinstance(config.params, FluxParams):
        raise ValueError(f"Config {mir_id} is not a flow model (expected FluxParams, got {type(config.params).__name__})")

    # Use provided repo_id/file_name or fall back to base spec
    checkpoint_repo_id = repo_id if repo_id is not None else config.repo_id
    checkpoint_file_name = file_name if file_name is not None else config.file_name

    # Create model
    with torch.device("meta"):
        if lora_repo_id and lora_filename:
            model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
        else:
            model = Flux(config.params).to(torch.bfloat16)

    # Load base checkpoint
    ckpt_path = str(retrieve_model(checkpoint_repo_id, checkpoint_file_name))
    nfo(f"Loading checkpoint: {ckpt_path}")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device=str(device))
    load_state_dict_into_model(model, sd, verbose=verbose)

    # Load LoRA if provided
    if lora_repo_id and lora_filename:
        if not isinstance(model, FluxLoraWrapper):
            raise ValueError("LoRA weights can only be loaded into FluxLoraWrapper models")
        load_lora_weights(model, lora_repo_id, lora_filename, device, verbose)

    return model


def load_t5(device: str | torch.device = device, max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = device) -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, dtype=torch.bfloat16).to(device)


def load_ae(mir_id: str, device: str | torch.device = device) -> AutoEncoder:
    """Load the autoencoder model.\n
    :param mir_id: Model ID (e.g., "model.vae.flux1-dev" or "model.taesd.flux1-dev")
    :param device: Device to load the model on
    :returns: Loaded AutoEncoder instance
    """
    config = get_model_spec(mir_id)

    if config.params is AutoencoderTiny:
        raise NotImplementedError("AutoencoderTiny loading not yet implemented. Use model.vae.flux1-dev instead.")

    if not isinstance(config.params, AutoEncoderParams):
        raise ValueError(f"Config {mir_id} is not an autoencoder (expected AutoEncoderParams, got {type(config.params).__name__})")

    ckpt_path = str(retrieve_model(config.repo_id, config.file_name))

    nfo("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.params)

    nfo(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))
    load_state_dict_into_model(ae, sd, verbose=True)
    return ae
