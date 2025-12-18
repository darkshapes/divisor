import math
import typing

import einops
import huggingface_hub
from nnll.init_gpu import device
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_flash_attention_utils import is_flash_attn_available

from divisor.contents import get_dtype
from divisor.duo.configuration_duo import DUOConfig
from divisor.flux1.layers import SelfAttention
from divisor.mmada.modeling_llada import BufferCache, RotaryEmbedding

precision = get_dtype(device)
if is_flash_attn_available():
    import flash_attn
    import flash_attn.layers.rotary


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float, training: bool
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_freqs(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    """
    Returns a tensor of shape (seq_len, head_dim, 2) that stores
    (cosine, sine) for each position / head dimension pair.
    The last dimension is split so that we can do a *real* complex multiplication
    without ever constructing a complex dtype tensor.
    """
    # Compute the base frequencies (θ = 10000^{-2i/head_dim})
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    # Position indices
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (seq_len, 1)

    # Outer product → angles (seq_len, head_dim/2)
    angles = pos * inv_freq  # broadcasting

    # 4Expand to full head_dim (cos, sin for even & odd indices)
    #    We interleave the even/odd dimensions so that the final shape is (seq_len, head_dim)
    cos = torch.cos(angles)  # (seq_len, head_dim/2)
    sin = torch.sin(angles)  # (seq_len, head_dim/2)

    # 5️⃣  Interleave to obtain (seq_len, head_dim)
    #    e.g. [c0, s0, c1, s1, …] → we need two separate tensors  for the real‑imag parts
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)  # (seq_len, head_dim)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)  # (seq_len, head_dim)

    # 6️⃣  Pack as (cos, sin) in the last dimension → shape (seq_len, head_dim, 2)
    freqs_cis = torch.stack([cos, sin], dim=-1)  # (seq_len, head_dim, 2)
    return freqs_cis


def apply_rotary_emb_from_cis(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k : (B, H, S, D)   – already reshaped for multi‑head attention
    freqs_cis : (S, D, 2) – (cos, sin) for each position & head‑dim pair

    Returns rotated (q, k) with the same shape.
    """
    # 1️⃣  Align dimensions: (1, 1, S, D, 2) so broadcasting works over B and H
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1,1,S,D,2)

    # 2️⃣  Split q/k into even/odd parts (real/imag) → shape (..., D,2)
    #     The last dimension will hold (real, imag) for each head‑dim pair.
    q_ = torch.stack([q[..., ::2], q[..., 1::2]], dim=-1)  #      (B,H,S,D/2,2)
    k_ = torch.stack([k[..., ::2], k[..., 1::2]], dim=-1)  #      (B,H,S,D/2,2)

    # 3️⃣  Perform the complex multiplication:
    #     (a + i·b) * (c + i·d) = (a·c - b·d) + i·(a·d + b·c)
    #     where (c,d) = (cos, sin) from freqs_cis.
    #     Because we duplicated cos/sin for both even/odd dims, we can just use the same
    #     freqs for every pair.
    cos = freqs[..., 0]  # (1,1,S,D,1)
    sin = freqs[..., 1]  # (1,1,S,D,1)

    # real part
    q_rot_real = q_[..., 0] * cos - q_[..., 1] * sin
    k_rot_real = k_[..., 0] * cos - k_[..., 1] * sin
    # imag part
    q_rot_imag = q_[..., 0] * sin + q_[..., 1] * cos
    k_rot_imag = k_[..., 0] * sin + k_[..., 1] * cos

    # 4️⃣  Re‑interleave back to the original shape (B,H,S,D)
    q_rot = torch.stack([q_rot_real, q_rot_imag], dim=-1).flatten(-2)  # (B,H,S,D)
    k_rot = torch.stack([k_rot_real, k_rot_imag], dim=-1).flatten(-2)  # (B,H,S,D)

    return q_rot, k_rot


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    if device.type == "cuda":
        context = torch.autocast(device_type=device.type, dtype=precision)
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        with context:
            cos, sin = rotary_cos_sin
            cos = cos.to(qkv.dtype)
            sin = sin.to(qkv.dtype)
            cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
            sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
            q, k, v = qkv.chunk(3, dim=2)
            if is_flash_attn_available():
                q = flash_attn.layers.rotary.apply_rotary_emb_torch(q.squeeze(dim=2), cos, sin)
                k = flash_attn.layers.rotary.apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
                v = v.squeeze(dim=2)
            else:
                xq_ = q.float().reshape(*q.shape[:-1], -1, 1, 2)
                xk_ = k.float().reshape(*k.shape[:-1], -1, 1, 2)
                xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
                xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
                return xq_out.reshape(*q.shape).type_as(q), xk_out.reshape(*k.shape).type_as(k)

                v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
    # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
    # where the 3 represents Q, K, V packed in that order
    attention_output = F.scaled_dot_product_attention(query=q.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False)
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, "b s h d -> b s (h d)")


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        if device.type == "cuda":
            context = torch.autocast(device_type=device.type, dtype=precision)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        x_skip = x
        if is_flash_attn_available():
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = self.norm1(x)
            qkv = self.attn_qkv(x)
            qkv = einops.rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
            with torch.autocast("cuda", enabled=False):
                cos, sin = rotary_cos_sin
                qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

            qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0.0, causal=True)

            x = einops.rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
        else:
            batch_size, seq_len = x.shape[:2]
            qkv = self.attn_qkv(x)
            qkv = einops.rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=self.n_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            freqs_cis = build_rotary_freqs(seq_len, self.n_heads * self.head_dim, device=x.device, dtype=x.dtype)
            q, k = apply_rotary_emb_from_cis(q, k, freqs_cis)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            x = einops.rearrange(attn_out, "b h s d -> b s (h d)")

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x, self.dropout)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN, cond_dim=None, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c=None):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = einops.rearrange(self.attn_qkv(x), "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

        x = regular_attention_multi_headed(q, k, v)

        if self.adaLN:
            x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
            x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum("blv,ve->ble", torch.nn.functional.softmax(x, dim=-1).float(), self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.model_config = DUOConfig()

        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)

        if is_flash_attn_available():
            self.rotary_emb = Rotary(dim // config.model.n_heads)
        else:
            self.__cache = BufferCache()
            self.rotary_emb = RotaryEmbedding(self.model_config, self.__cache)

        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout)
            else:
                block = DDiTBlock(dim=dim, n_heads=config.model.n_heads, cond_dim=cond_dim, adaLN=self.adaLN, dropout=config.model.dropout)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(hidden_size=dim, out_channels=vocab_size, cond_dim=cond_dim, adaLN=self.adaLN)
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma):
        x = self.vocab_embed(x)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x).to(x.device)

        if device.type == "cuda":
            context = torch.autocast(device_type=device.type, dtype=precision)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
            x = self.output_layer(x, c=t_cond)

        return x


class HFDIT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = config.causal
        self.adaLN = not self.causal
        self.vocab_size = config.vocab_size
        dim = config.hidden_dim
        cond_dim = config.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, self.vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)

        blocks = []
        for _ in range(config.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(dim=dim, n_heads=config.n_heads, dropout=config.dropout)
            else:
                block = DDiTBlock(dim=dim, n_heads=config.n_heads, cond_dim=cond_dim, adaLN=self.adaLN, dropout=config.dropout)
            blocks.append(block)
        self.blocks = torch.nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(hidden_size=dim, out_channels=self.vocab_size, cond_dim=cond_dim, adaLN=self.adaLN)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, sigma, output_hidden_states=False):
        all_hidden_states = []
        x = self.vocab_embed(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        if device.type == "cuda":
            context = torch.autocast(device_type=device.type, dtype=precision)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
                if output_hidden_states:
                    all_hidden_states.append(x)
            x = self.output_layer(x, c=t_cond)
        return x, all_hidden_states


class DUO(transformers.PreTrainedModel):
    """HF-compatible model."""

    config_class = DUOConfig
    base_model_prefix = "duo"

    def __init__(self, config: DUOConfig):
        super().__init__(config)
        self.config = config
        self.backbone = HFDIT(config)

    def reset_kv_cache(self):
        for block in self.backbone.blocks:
            block.kv_cache = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        timesteps: torch.FloatTensor = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
    ) -> typing.Union[torch.Tensor, typing.Tuple, transformers.modeling_outputs.MaskedLMOutput]:
        """HF-compatible forward method."""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits, all_hidden_states = self.backbone(
            x=input_ids,
            sigma=timesteps,
            output_hidden_states=output_hidden_states,
        )
        if return_dict:
            return transformers.modeling_outputs.MaskedLMOutput(logits=logits, hidden_states=all_hidden_states if output_hidden_states else None, loss=None)
        elif output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits
