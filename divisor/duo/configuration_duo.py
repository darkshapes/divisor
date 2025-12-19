# SPDX-License-Identifier: Apache-2.0
# adapted from https://github.com/s-sahoo/duo

from transformers import PretrainedConfig, AutoConfig


class DUOConfig(PretrainedConfig):
    """Hugging Face configuration class for DUO."""

    model_type = "DUO"

    def __init__(
        self,
        vocab_size: int = 50258,
        model_length: int = 1024,
        model_type: str = "DUO",
        causal: bool = False,
        hidden_dim: int = 768,
        cond_dim: int = 129,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        var_min: bool = True,
        rope_theta: float = 10000.0,
        max_sequence_length: int = 1024,
        init_device: str | None = None,
        d_model: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.causal = causal
        self.vocab_size = vocab_size
        self.model_length = model_length
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.var_min = var_min
        self.model_type = model_type
        self.rope_theta = rope_theta
        self.max_sequence_length = max_sequence_length
        self.init_device = init_device
        self.d_model = d_model
        for key, value in kwargs.items():
            setattr(self, key, value)


AutoConfig.register("DUO", DUOConfig)
