# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest

from divisor.flux1.spec import configs as flux_configs
from divisor.mmada.spec import configs as mmada_configs
from divisor.spec import populate_model_choices, get_model_spec


def test_populate_model_choices():
    """Test that populate_model_choices returns the correct model choices."""
    flux_models = populate_model_choices(flux_configs)
    mmada_models = populate_model_choices(mmada_configs)
    assert len(flux_models) > 0
    assert len(mmada_models) > 0
    assert "model.dit.flux1-dev" in flux_models
    assert "model.mldm.mmada" in mmada_models
    assert "model.dit.flux1-dev:mini" in flux_models


def test_populate_model_choice_that_doesnt_exist():
    """Test that populate_model_choices returns the correct model choices."""
    flux_models = populate_model_choices(flux_configs)
    mmada_models = populate_model_choices(mmada_configs)
    assert len(flux_models) > 0
    assert len(mmada_models) > 0
    assert "model.mldm.mmada:base" not in mmada_models
    assert "model.dit.flux_drop_da:base" not in flux_models


def test_get_model_spec_flux():
    """Test that get_model_spec returns the correct model spec."""
    from divisor.flux1.spec import configs

    flux_models = populate_model_choices(configs)
    model_spec = get_model_spec(flux_models[0], configs)
    assert model_spec is not None
    assert model_spec.repo_id == "black-forest-labs/FLUX.1-dev"
    assert model_spec.file_name == "flux1-dev.safetensors"
    assert model_spec.params is not None
    assert model_spec.init is not None


def test_get_model_spec_flux_sub_key():
    """Test that get_model_spec returns the correct model spec."""
    from divisor.flux1.spec import configs

    flux_models = populate_model_choices(flux_configs)
    model_spec = get_model_spec(flux_models[2], configs)
    assert model_spec is not None
    assert model_spec.repo_id == "Kijai/flux-fp8"
    assert model_spec.file_name == "flux1-dev-fp8-e4m3fn.safetensors"
    assert model_spec.params is not None
    assert model_spec.init is not None


def test_get_model_spec_flux_sub_key_that_doesnt_exist():
    """Test that get_model_spec returns the correct model spec."""
    from divisor.flux1.spec import configs

    flux_models = populate_model_choices(flux_configs)
    with pytest.raises(ValueError, match="doesnt-exist has no defined model spec"):
        model_spec = get_model_spec(flux_models[0] + ":doesnt-exist", configs)


def test_get_model_spec_mmada():
    """Test that get_model_spec returns the correct model spec."""
    from divisor.mmada.spec import configs

    mmada_models = populate_model_choices(mmada_configs)
    print(mmada_models)
    print(list(configs))
    model_spec = get_model_spec(mmada_models[0], configs)
    assert model_spec is not None
    assert model_spec.repo_id == "Gen-Verse/MMaDA-8B-Base"
    assert model_spec.params is not None
    assert model_spec.init is not None
