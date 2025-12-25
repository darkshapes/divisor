import json
from pathlib import Path
import random


DEFAULT_ROOT_DIR = "examples/default/input_params"
ZH_RAP_LORA_ROOT_DIR = "examples/zh_rap_lora/input_params"


class DataSampler:
    def __init__(self, root_dir=DEFAULT_ROOT_DIR):
        self.root_dir_path = Path(root_dir).resolve()
        self.root_dir = str(self.root_dir_path)
        self.input_params_files = list(self.root_dir_path.glob("*.json"))
        self.zh_rap_lora_input_params_files = list(Path(ZH_RAP_LORA_ROOT_DIR).glob("*.json"))
        self.zh_rap_lora_input_params_files += list(Path(ZH_RAP_LORA_ROOT_DIR).glob("*.json"))

    def load_json(self, file_path):
        """Load a JSON file located under this sampler's root directory.
        The given file_path may be a simple filename or a relative path, but it is
        always resolved relative to self.root_dir_path and must not escape that root.
        interpret the incoming path relative to the configured root directory.
        This prevents callers from supplying absolute paths or traversing outside
        of self.root_dir_path using constructs like "../"."""

        file_path_obj = Path(file_path)
        if file_path_obj.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: {file_path}")
        combined_path = (self.root_dir_path / file_path_obj).resolve()
        try:
            combined_path.relative_to(self.root_dir_path)
        except ValueError:
            raise ValueError(f"Access to file outside of root directory is not allowed: {file_path}")
        with combined_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def sample(self, lora_name_or_path=None):
        if lora_name_or_path is None or lora_name_or_path == "none":
            json_path = random.choice(self.input_params_files)
            json_data = self.load_json(json_path)
        else:
            json_path = random.choice(self.zh_rap_lora_input_params_files)
            json_data = self.load_json(json_path)
            # Update the lora_name in the json_data
            json_data["lora_name_or_path"] = lora_name_or_path

        return json_data
