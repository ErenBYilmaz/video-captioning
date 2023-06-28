import os.path
import shutil
from typing import Literal

from data.image_metadata import ImageMetadata


class NamedImageMetadata(ImageMetadata):
    def __init__(self, image_type: Literal['png', 'jpg'], name: str):
        super().__init__(image_type=image_type)
        self.name = name

    def base_file_name(self):
        return self.name
