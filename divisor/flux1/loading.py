# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux


import os
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_sft
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from nnll.console import nfo
from nnll.init_gpu import device

from divisor.flux1.spec import optionally_expand_state_dict, get_model_spec
from divisor.flux1.model import FluxLoraWrapper, Flux, FluxParams
from divisor.flux1.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux1.text_embedder import HFEmbedder


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        nfo(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        nfo("\n" + "-" * 79 + "\n")
        nfo(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        nfo(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        nfo(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def retrieve_model(repo_id: str, file_name: str) -> Path:
    """Get the local path for a checkpoint file, downloading if necessary.
    :param repo_id: Repository ID for the checkpoint
    :param file_name: Name of the checkpoint file
    :returns: Path to the checkpoint file
    """

    model_dir = snapshot_download(repo_id=repo_id, allow_patterns=[file_name])
    return Path(model_dir) / file_name


def convert_fp8_to_bf16(model: torch.nn.Module, verbose: bool = True) -> bool:
    """Detect and convert fp8 tensors in model parameters and buffers to bf16.
    :param model: The model to convert fp8 tensors in
    :param verbose: Whether to print conversion messages
    """
    # Define fp8 dtypes to detect
    fp8_dtypes = [
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ]

    fp8_dtypes = [dtype for dtype in fp8_dtypes if dtype is not None]
    if not fp8_dtypes:
        return False

    converted_count = 0

    for name, param in model.named_parameters():
        if param.dtype in fp8_dtypes:
            with torch.no_grad():
                param.data = param.data.float()
            converted_count += 1
            if verbose:
                nfo(f"Converted fp8 parameter '{name}' to bf16")

    for name, buffer in model.named_buffers():
        if buffer.dtype in fp8_dtypes:
            buffer.data = buffer.data.float()
            converted_count += 1
            if verbose:
                nfo(f"Converted fp8 buffer '{name}' to bf16")

    if converted_count > 0:
        nfo(f"Converted {converted_count} fp8 tensor(s) to bf16")
        return True
    return False


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
    device: torch.device = device,
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
    nfo(f": {os.path.basename(ckpt_path)}")
    sd = load_sft(ckpt_path, device=device.type)

    load_state_dict_into_model(model, sd, verbose=verbose)

    if device.type == "mps":
        convert_fp8_to_bf16(model, verbose=verbose)

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

    with torch.device("meta"):
        ae = AutoEncoder(config.params)

    nfo(f": {os.path.basename(ckpt_path)}")
    sd = load_sft(ckpt_path, device=device.type)
    load_state_dict_into_model(ae, sd, verbose=True)
    return ae
