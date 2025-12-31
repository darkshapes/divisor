# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from nnll.save_generation import name_save_file_as, save_with_metadata
from nnll.constants import ExtensionType
from nnll.hyperchain import HyperChain
from PIL import Image


class SaveFile:
    intermediate_image: Image
    hyperchain: HyperChain
    extension: ExtensionType = ExtensionType.WEBP
    save_folder_path: Path = Path(__file__).resolve().parent.parent / ".output"

    def __enter__(self) -> "SaveFile":
        """Return instanced self"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Perform cleanup regardless of success or exception"""
        self._cleanup()

    def _cleanup(self) -> None:
        """Release resources held by the instance and close methods (PIL.Image)"""
        if hasattr(self.intermediate_image, "close"):
            try:
                self.intermediate_image.close()
            except Exception:
                pass  # ignore errors during cleanup

        self.intermediate_image = None  # type: ignore[assignment]
        self.hyperchain = None  # type: ignore[assignment]

    def with_hyperchain(self) -> None:
        file_path_named = name_save_file_as(
            extension=self.extension,
            save_folder_path=self.save_folder_path,
        )
        save_with_metadata(file_path_named, self.intermediate_image, str(self.hyperchain))
