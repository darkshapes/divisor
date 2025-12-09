# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import replace
from typing import Any, List, Tuple

from nnll.init_gpu import device
import torch


def get_dtype(device: torch.device = device) -> torch.dtype:
    dtype_by_device = {
        "cuda": torch.bfloat16,
        "mps": torch.bfloat16,
        "cpu": torch.float32,
    }
    return dtype_by_device[device.type]


def find_mir_spec(
    model_id: str,
    ae_id: str,
    configs: dict,
    tiny: bool = False,
    prefix: str = "model.dit.",
) -> Tuple[str, str | None, str]:
    """Find/validate model specifications by MIR (Machine Intelligence Resource) ID.\n
    :param model_id: Model ID, optionally with subkey (e.g., "flux1-dev" or "flux1-dev:mini")
    :param ae_id: Autoencoder ID
    :param configs: Configuration dictionary containing model specs
    :param tiny: Whether to use tiny autoencoder prefix (model.taesd. instead of model.vae.)
    :param prefix: Prefix to add to model_id (default: "model.dit.")
    :returns: Tuple of (normalized_model_id, subkey, normalized_ae_id)
    :raises ValueError: If model_id, subkey, or ae_id is not found in configs
    """

    def _validate_in_configs(key: str, key_type: str, available: List[str] | None = None) -> None:
        """Helper to validate a key exists in configs."""
        if key not in configs:
            available_keys = available if available is not None else list(configs.keys())
            available_str = ", ".join(available_keys)
            raise ValueError(f"Got unknown {key_type}: {key}, chose from {available_str}")

    # Handle model_id with optional subkey
    subkey = None
    if ":" in model_id:
        base_model_id, subkey = model_id.split(":", 1)
        normalized_model_id = f"{prefix}{base_model_id}".lower()
        subkey = subkey.lower()
        _validate_in_configs(
            normalized_model_id,
            "base model id",
        )
        base_model = configs[normalized_model_id]

        if subkey not in base_model:
            available_subkeys = [k for k in base_model.keys() if k != "*"]
            available_str = ", ".join(available_subkeys)
            raise ValueError(f"Got unknown subkey '{subkey}' for model {normalized_model_id}. Available subkeys: {available_str}")

        base_spec = base_model["*"]
        subkey_spec = base_model[subkey]

        # Merge subkey spec with base spec (subkey values take precedence)
        merged_spec = merge_spec(base_spec, subkey_spec)
        if merged_spec is not base_spec:
            # Update configs with merged spec so subsequent lookups use merged values
            configs[normalized_model_id] = {**base_model, "*": merged_spec}
    else:
        normalized_model_id = f"{prefix}{model_id}".lower()
        _validate_in_configs(normalized_model_id, "model id")

    ae_prefix = "model.taesd." if tiny else "model.vae."
    normalized_ae_id = f"{ae_prefix}{ae_id}".lower()
    _validate_in_configs(normalized_ae_id, "ae id")

    return normalized_model_id, subkey, normalized_ae_id


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


def get_model_spec(mir_id: str, configs: list[dict[str, Any]]) -> Any | None:
    """Get a ModelSpec or CompatibilitySpec for a given model ID. Use to point to a known model spec.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param configs: Configuration mapping containing model specs
    :returns: ModelSpec if compatibility_key is None, CompatibilitySpec if provided and available, None if provided but not found
    :raises ValueError: If model ID does not have a base ModelSpec
    """

    if ":" in mir_id:
        spec_key, compatibility_key = mir_id.split(":")
        for config_entry in configs:
            if spec_key in config_entry:
                base_entry = config_entry[spec_key]["*"]
                if compat_entry := config_entry[spec_key].get(compatibility_key, None):
                    return merge_spec(base_entry, compat_entry)
                raise ValueError(f"{mir_id} has no defined model spec")
    else:
        for config_entry in configs:
            if spec_entry := config_entry.get(mir_id, None):
                return spec_entry["*"]
            raise ValueError(f"{mir_id} has no defined model spec")
