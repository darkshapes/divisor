# SPDX-License-Identifier:Apache-2.0
# original XFlux code from https://github.com/TencentARC/FluxKits

from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor
from einops import rearrange
import uuid
import os

from nnll.save_generation import save_with_hyperchain

from divisor.xflux1.layers import (
    SingleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from divisor.xflux1.sampling import denoise
from divisor.flux1.sampling import get_noise, get_schedule, prepare, unpack
from divisor.flux1.loading import load_ae, load_clip, load_flow_model, load_t5
from divisor.xflux1.loading import get_lora_rank, load_checkpoint


class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }

    def set_lora(self, local_path: str = None, repo_id: str = None, name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(None, self.hf_lora_collection, self.lora_types_to_names[lora_type])
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            if name.startswith("single_blocks"):
                lora_attn_procs[name] = SingleStreamBlockProcessor()  # SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                continue
            elif name.startswith("double_blocks"):
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)

            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1 :]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict, strict=False)
            lora_attn_procs[name].to(self.device)

        self.model.set_attn_processor(lora_attn_procs)

    def __call__(
        self,
        prompt: str,
        image_prompt: Image.Image | None = None,
        controlnet_image: Image.Image | None = None,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 3,
        control_weight: float = 0.9,
        ip_scale: float = 1.0,
        neg_ip_scale: float = 1.0,
        neg_prompt: str = "",
        neg_image_prompt: Image.Image | None = None,
        timestep_to_start_cfg: int = 0,
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )

    @torch.inference_mode()
    def gradio_generate(
        self,
        prompt,
        image_prompt,
        controlnet_image,
        width,
        height,
        guidance,
        num_steps,
        seed,
        true_gs,
        ip_scale,
        neg_ip_scale,
        neg_prompt,
        neg_image_prompt,
        timestep_to_start_cfg,
        control_type,
        control_weight,
        lora_weight,
        local_path,
        lora_local_path,
        ip_local_path,
    ):
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(
            prompt,
            image_prompt,
            controlnet_image,
            width,
            height,
            guidance,
            num_steps,
            seed,
            true_gs,
            control_weight,
            ip_scale,
            neg_ip_scale,
            neg_prompt,
            neg_image_prompt,
            timestep_to_start_cfg,
        )

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image=None,
        timestep_to_start_cfg=0,
        true_gs=3.5,
        control_weight=0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(1, height, width, device=self.device, dtype=torch.bfloat16, seed=seed)
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond["txt"],
                neg_txt_ids=neg_inp_cond["txt_ids"],
                neg_vec=neg_inp_cond["vec"],
                true_gs=true_gs,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload:
            return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
