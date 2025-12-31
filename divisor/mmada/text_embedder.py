# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA
# Adapted from https://github.com/black-forest-labs/flux

from divisor.registry import device
import torch
from transformers import AutoTokenizer

from divisor.mmada.prompting_utils import UniversalPrompting, reserved_token_mapping


class HFEmbedder:
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        # self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        self.tokenizer = AutoTokenizer.from_pretrained(version, trust_remote_code=True)
        self.max_length = max_length

    def forward(self, text: list[str], modality: str, image_tokens: torch.Tensor | None = None) -> dict:
        image_tokens = torch.ones((1, 1024), dtype=torch.long, device=device) * self.mask_id

        if modality == "t2i":
            if hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
                self.mask_id = self.tokenizer.mask_token_id
            else:
                self.mask_id = 126336

            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            prompt_text = [text]
            uni_prompting = UniversalPrompting(
                self.tokenizer,
                max_text_len=self.max_length,
                special_tokens=(list(reserved_token_mapping)[:-2]),
                ignore_id=-100,
                cond_dropout_prob=0.1,
                use_reserved_token=True,
            )
            self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
            input_ids, attention_mask = uni_prompting((prompt_text, image_tokens), "t2i_gen")  # type: ignore
            outputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mask_id": self.mask_id,
                "uni_prompting": uni_prompting,
            }
        else:
            template = [{"role": "user", "content": text}]
            processed_prompt_text = self.tokenizer.apply_chat_template(template, add_generation_prompt=True, tokenize=False)
            input_ids = self.tokenizer(
                text=processed_prompt_text,
                return_tensors="pt",
                padding="longest",
                padding_side="left",
                truncation=True,
                max_length=self.max_length,
            )["input_ids"].to(device)

        return outputs  # type: ignore [self.output_key]
