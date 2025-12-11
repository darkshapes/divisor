"""Tests for find_mir_spec and merge_spec functions."""

import pytest
from copy import deepcopy

from divisor.spec import merge_spec, get_model_spec
from divisor.flux1.spec import configs as flux1_configs, ModelSpec, InitialParams
from divisor.contents import build_available_models
from divisor.mmada.spec import configs as mmada_configs


# class TestFindMirSpec:
#     """Test find_mir_spec function operation."""

#     def test_find_mir_spec_with_simple_model_id(self):
#         """Test find_mir_spec with a simple model ID (no subkey)."""
#         # Use a copy of configs to avoid modifying the original
#         test_configs = deepcopy(configs)

#         flux_args = build_available_models(flux1_configs)
#         mmada_args = build_available_models(mmada_configs)
#         model_args = flux_args | mmada_args
#         model_id = get_model_spec("flux1-dev", flux1_configs)

#         assert model_id == "model.dit.flux1-dev"
#         assert subkey is None


#     def test_find_mir_spec_with_subkey(self):
#         """Test find_mir_spec with model ID containing subkey."""
#         test_configs = deepcopy(configs)

#         model_id, subkey, ae_id = find_mir_spec(
#             model_id="flux1-dev:mini",
#             ae_id="flux1-dev",
#             configs=test_configs,
#         )

#         assert model_id == "model.dit.flux1-dev"
#         assert subkey == "mini"
#         assert ae_id == "model.vae.flux1-dev"

#         # Verify that the merged spec was stored in configs
#         merged_spec = test_configs[model_id]["*"]
#         assert isinstance(merged_spec, ModelSpec)
#         # Verify subkey values took precedence
#         assert merged_spec.repo_id == "TencentARC/flux-mini"  # From mini subkey
#         assert merged_spec.file_name == "flux-mini.safetensors"  # From mini subkey

#     def test_find_mir_spec_with_tiny_ae(self):
#         """Test find_mir_spec with tiny autoencoder."""
#         test_configs = deepcopy(configs)

#         model_id, subkey, ae_id = find_mir_spec(
#             model_id="flux1-dev",
#             ae_id="flux1-dev",
#             configs=test_configs,
#             tiny=True,
#         )

#         assert ae_id == "model.taesd.flux1-dev"

#     def test_find_mir_spec_raises_error_for_invalid_model_id(self):
#         """Test that find_mir_spec raises ValueError for invalid model ID."""
#         test_configs = deepcopy(configs)

#         with pytest.raises(ValueError, match="Got unknown model id"):
#             find_mir_spec(
#                 model_id="invalid-model",
#                 ae_id="flux1-dev",
#                 configs=test_configs,
#             )

#     def test_find_mir_spec_raises_error_for_invalid_base_model(self):
#         """Test that find_mir_spec raises ValueError for invalid base model in subkey format."""
#         test_configs = deepcopy(configs)

#         with pytest.raises(ValueError, match="Got unknown base model id"):
#             find_mir_spec(
#                 model_id="invalid-base:mini",
#                 ae_id="flux1-dev",
#                 configs=test_configs,
#             )

#     def test_find_mir_spec_raises_error_for_invalid_subkey(self):
#         """Test that find_mir_spec raises ValueError for invalid subkey."""
#         test_configs = deepcopy(configs)

#         with pytest.raises(ValueError, match="Got unknown subkey"):
#             find_mir_spec(
#                 model_id="flux1-dev:invalid-subkey",
#                 ae_id="flux1-dev",
#                 configs=test_configs,
#             )

#     def test_find_mir_spec_raises_error_for_invalid_ae_id(self):
#         """Test that find_mir_spec raises ValueError for invalid AE ID."""
#         test_configs = deepcopy(configs)

#         with pytest.raises(
#             ValueError, match="Got unknown ae id: model.vae.invalid-ae, chose from model.dit.flux1-dev, model.vae.flux1-dev, model.taesd.flux1-dev, model.dit.flux1-schnell"
#         ):
#             find_mir_spec(
#                 model_id="flux1-dev",
#                 ae_id="invalid-ae",
#                 configs=test_configs,
#             )

#     def test_find_mir_spec_merges_nested_dataclasses(self):
#         """Test that find_mir_spec correctly merges nested dataclasses (e.g., init)."""
#         test_configs = deepcopy(configs)

#         model_id, subkey, ae_id = find_mir_spec(
#             model_id="flux1-dev:mini",
#             ae_id="flux1-dev",
#             configs=test_configs,
#         )

#         # Get the merged spec
#         merged_spec = test_configs[model_id]["*"]
#         assert isinstance(merged_spec, ModelSpec)
#         assert merged_spec.init is not None

#         # Verify nested dataclass (init) was merged correctly
#         # Base flux1-dev has num_steps=28, mini has num_steps=25
#         # Mini's value should take precedence
#         assert merged_spec.init.num_steps == 25  # From mini subkey
#         assert merged_spec.init.guidance == 3.5  # From mini subkey
#         assert merged_spec.init.shift is True  # From both, but mini's value takes precedence


# class TestMergeSpec:
#     """Test merge_spec function operation."""

#     def test_merge_spec_with_simple_fields(self):
#         """Test merge_spec with simple (non-nested) fields."""
#         base_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name="base.safetensors",
#             params=None,
#             init=None,
#         )
#         subkey_spec = ModelSpec(
#             repo_id="subkey/repo",
#             file_name="subkey.safetensors",
#             params=None,
#             init=None,
#         )

#         merged = merge_spec(base_spec, subkey_spec)

#         assert merged.repo_id == "subkey/repo"  # Subkey takes precedence
#         assert merged.file_name == "subkey.safetensors"  # Subkey takes precedence
#         assert merged is not base_spec  # Should be a new instance

#     def test_merge_spec_with_nested_dataclasses(self):
#         """Test merge_spec with nested dataclasses (init field)."""
#         base_init = InitialParams(
#             num_steps=28,
#             max_length=512,
#             guidance=4.0,
#             shift=True,
#         )
#         subkey_init = InitialParams(
#             num_steps=25,
#             max_length=512,  # Same as base
#             guidance=3.5,
#             shift=True,  # Same as base
#         )

#         base_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name="base.safetensors",
#             params=None,
#             init=base_init,
#         )
#         subkey_spec = ModelSpec(
#             repo_id="base/repo",  # Same as base
#             file_name="base.safetensors",  # Same as base
#             params=None,
#             init=subkey_init,
#         )

#         merged = merge_spec(base_spec, subkey_spec)

#         assert merged.init is not None
#         assert merged.init.num_steps == 25  # Subkey value
#         assert merged.init.max_length == 512  # Subkey value (same as base)
#         assert merged.init.guidance == 3.5  # Subkey value
#         assert merged.init.shift is True  # Subkey value (same as base)

#     def test_merge_spec_partial_override(self):
#         """Test merge_spec when subkey only overrides some fields."""
#         base_init = InitialParams(
#             num_steps=28,
#             max_length=512,
#             guidance=4.0,
#             shift=True,
#         )
#         # Subkey only overrides num_steps, others should remain from base
#         subkey_init = InitialParams(
#             num_steps=25,
#             max_length=512,
#             guidance=4.0,
#             shift=True,
#         )

#         base_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name="base.safetensors",
#             params=None,
#             init=base_init,
#         )
#         subkey_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name="base.safetensors",
#             params=None,
#             init=subkey_init,
#         )

#         merged = merge_spec(base_spec, subkey_spec)

#         assert merged.init.num_steps == 25  # Overridden by subkey
#         assert merged.init.guidance == 4.0  # From subkey (same as base)

#     def test_merge_spec_returns_base_if_not_dataclass(self):
#         """Test merge_spec returns base if subkey is not a dataclass."""
#         base_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name="base.safetensors",
#             params=None,
#             init=None,
#         )
#         subkey_spec = "not a dataclass"

#         merged = merge_spec(base_spec, subkey_spec)

#         assert merged is base_spec  # Should return base unchanged

#     def test_merge_spec_handles_none_values(self):
#         """Test merge_spec handles None values in subkey correctly."""
#         base_spec = ModelSpec(
#             repo_id="base/repo",
#             file_name=None,
#             params=None,
#             init=InitialParams(num_steps=28, max_length=512, guidance=4.0, shift=True),
#         )
#         subkey_spec = ModelSpec(
#             repo_id="subkey/repo",
#             file_name=None,  # None value
#             params=None,
#             init=None,  # None value
#         )

#         merged = merge_spec(base_spec, subkey_spec)

#         assert merged.repo_id == "subkey/repo"  # Subkey value
#         assert merged.file_name is None  # None from subkey
#         assert merged.init is not None  # Should keep base value since subkey is None
#         assert merged.init.num_steps == 28  # From base
