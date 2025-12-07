# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import json
import os
from typing import Callable, Any
from huggingface_hub import snapshot_download


def create_config_reader(model_id: str = "Gen-Verse/MMaDA-8B-Base") -> Callable[[str], Any]:
    cache_folder_named = snapshot_download(model_id)
    config_file = os.path.join(cache_folder_named, "config.json")

    def read_config(key_name: str):
        with open(config_file, "r") as f:
            config_data = json.load(f)
        return config_data.get(key_name, None)

    return read_config
