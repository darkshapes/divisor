# SPDX-License-Identifier:Apache-2.0
# adapted from https://github.com/ace-step/ACE-Step

import torch
import functools
from typing import Callable, TypeVar
from divisor.registry import gfx


class CpuOffloader:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.original_device = device
        self.original_dtype = model.dtype

    def __enter__(self):
        if not hasattr(self.model, "torchao_quantized"):
            self.model.to(self.original_device, dtype=self.original_dtype)
        return self.model

    def __exit__(self, *args):
        if not hasattr(self.model, "torchao_quantized"):
            self.model.to("cpu")
        gfx.sync()
        gfx.cache()


T = TypeVar("T")


def cpu_offload(model_attr: str):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.cpu_offload:
                return func(self, *args, **kwargs)

            # Get the device from the class
            device = self.device
            # Get the model from the class attribute
            model = getattr(self, model_attr)

            with CpuOffloader(model, device):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
