# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Any


def populate_model_choices(configs: dict[str, Any]) -> list[str]:
    """Generate model choices from all entries in configs\n
    :returns: List of model choices"""
    model_choices = []
    for model_id, model_config in configs.items():
        model_choices.append(model_id)
        model_choices.extend([f"{model_id}:{n}" for n in model_config.keys() if n != "*"])
    return model_choices


def build_available_models(configs: dict[str, Any]) -> dict[str, str]:
    """Build model arguments from configs.\n
    :param configs: Configuration mapping containing model specs
    :returns: List of model arguments
    """

    model_choices = populate_model_choices(configs)
    model_args: dict = {}
    filters = ["fp8-", ".vae."]
    for model in model_choices:
        if not any(filter in model for filter in filters):
            if ":" in model:
                key = model.split(":")[-1]
            else:
                key = model.split(".")[-1]
            if key not in model_args:
                model_args[key] = model

    return model_args
