# SPDX-License-Identifier:Apache-2.0
# original BFL Flux code from https://github.com/black-forest-labs/flux

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_sft
from nnll.init_gpu import device

from divisor.flux_modules.model import Flux, FluxParams, FluxLoraWrapper
from divisor.flux_modules.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux_modules.text_embedder import HFEmbedder

CHECKPOINTS_DIR = Path(snapshot_download(repo_id="black-forest-labs/FLUX.1-schnell", local_files_only=False))
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def get_checkpoint_path(repo_id: str, filename: str, env_var: str) -> Path:
    """Get the local path for a checkpoint file, downloading if necessary.

    :param repo_id: Repository ID for the checkpoint
    :param filename: Name of the checkpoint file
    :param env_var: Environment variable name to check for custom path
    :returns: Path to the checkpoint file
    """
    if os.environ.get(env_var) is not None:
        local_path = os.environ[env_var]
        if os.path.exists(local_path):
            return Path(local_path)

        print(f"Trying to load model {repo_id}, {filename} from environment variable {env_var}. But file {local_path} does not exist. Falling back to default location.")

    # Create a safe directory name from repo_id
    safe_repo_name = repo_id.replace("/", "_")
    checkpoint_dir = CHECKPOINTS_DIR / safe_repo_name
    checkpoint_dir.mkdir(exist_ok=True)

    local_path = checkpoint_dir / filename

    return local_path


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    repo_id: str
    repo_flow: str
    repo_ae: str
    lora_repo_id: str | None = None
    lora_filename: str | None = None


configs = {
    "flux-dev": ModelSpec(
        repo_id="",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-krea": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        repo_flow="flux1-krea-dev.safetensors",
        repo_ae="ae.safetensors",
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
        ae_params=AutoEncoderParams(
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
    "flux-schnell": ModelSpec(
        repo_id="",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Canny-dev-lora",
        lora_filename="flux1-canny-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Depth-dev-lora",
        lora_filename="flux1-depth-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-redux": ModelSpec(
        repo_id="cocktailpeanut/redux",
        repo_flow="flux1-redux-dev.safetensors",
        repo_ae="ae.safetensors",
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-fill": ModelSpec(
        repo_id="cocktailpeanut/fill",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=384,
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
        ae_params=AutoEncoderParams(
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
    "flux-dev-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
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
        ae_params=AutoEncoderParams(
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
}


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """Optionally expand the state dict to match the model's parameters shapes.

    :param model: The model to match parameters against
    :param state_dict: The state dictionary to expand
    :returns: The expanded state dictionary
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}.")
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict


def aspect_ratio_to_height_width(aspect_ratio: str, area: int = 1024**2) -> tuple[int, int]:
    width = float(aspect_ratio.split(":")[0])
    height = float(aspect_ratio.split(":")[1])
    ratio = width / height
    width = round(math.sqrt(area * ratio))
    height = round(math.sqrt(area / ratio))
    return 16 * (width // 16), 16 * (height // 16)


def load_flow_model(name: str, device: str | torch.device = device, verbose: bool = True) -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]

    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
        else:
            model = Flux(config.params).to(torch.bfloat16)

    print(f"Loading checkpoint: {ckpt_path}")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device=str(device))
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)

    if config.lora_repo_id is not None and config.lora_filename is not None:
        print("Loading LoRA")
        lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
        lora_sd = load_sft(lora_path, device=str(device))
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = device, max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = device) -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = device) -> AutoEncoder:
    config = configs[name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE"))

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae
