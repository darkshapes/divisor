# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import replace
from typing import Callable, Any

from nnll.init_gpu import device
import torch


def get_dtype(device: torch.device = device) -> torch.dtype:
    dtype_by_device = {
        "cuda": torch.bfloat16,
        "mps": torch.bfloat16,
        "cpu": torch.float32,
    }
    return dtype_by_device[device.type]


def merge_spec(base_spec: Any, subkey_spec: Any) -> Any:
    """Merge two dataclass or nested dataclass specs with overlapping subkey values taking precedence over base values.\n
    :param base_spec: Base specification dataclass
    :param subkey_spec: Subkey specification dataclass (values take precedence)
    :returns: Merged specification with subkey values overriding base values
    """
    if not hasattr(subkey_spec, "__dataclass_fields__"):
        return base_spec

    merge_kwargs = {}
    for field_name in subkey_spec.__dataclass_fields__:
        subkey_value = getattr(subkey_spec, field_name, None)
        base_value = getattr(base_spec, field_name, None)

        if subkey_value is not None:
            # Handle nested dataclasses (e.g., init: InitialParams)
            if hasattr(subkey_value, "__dataclass_fields__") and base_value is not None and hasattr(base_value, "__dataclass_fields__"):
                # Merge nested dataclass: subkey fields override base fields
                nested_merge_kwargs = {}
                for nested_field in subkey_value.__dataclass_fields__:
                    nested_subkey_val = getattr(subkey_value, nested_field, None)
                    if nested_subkey_val is not None:
                        nested_merge_kwargs[nested_field] = nested_subkey_val
                if nested_merge_kwargs:
                    merge_kwargs[field_name] = replace(base_value, **nested_merge_kwargs)
                else:
                    merge_kwargs[field_name] = subkey_value
            else:
                merge_kwargs[field_name] = subkey_value

    if merge_kwargs:
        return replace(base_spec, **merge_kwargs)
    return base_spec


def get_model_spec(mir_id: str, configs: dict[str, dict[str, Callable]]) -> Callable | None:
    """Get a ModelSpec or CompatibilitySpec for a given model ID. Use to point to a known model spec.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param configs: Configuration mapping containing model specs
    :returns: ModelSpec if compatibility_key is None, CompatibilitySpec if provided and available, None if provided but not found
    :raises ValueError: If model ID does not have a base ModelSpec
    """

    if ":" in mir_id:
        series_key, compatibility_key = mir_id.split(":")
        for config_entry in configs:
            if series_key in config_entry:
                base_entry = config_entry[series_key]["*"]
                if compat_entry := config_entry[series_key].get(compatibility_key, None):
                    return merge_spec(base_entry, compat_entry)
                raise ValueError(f"{mir_id} has no defined model spec")
    else:
        for config_entry in configs:
            if spec_entry := config_entry.get(mir_id, None):
                return spec_entry["*"]
            raise ValueError(f"{mir_id} has no defined model spec")
