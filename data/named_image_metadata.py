from typing import Dict, Any

from data.image_metadata import ImageMetadata


class NamedImageMetadata(ImageMetadata):
    def __init__(self, image_type: str, name: str, tool_outputs: Dict[str, Any] = None):
        super().__init__(image_type=image_type, tool_outputs=tool_outputs)
        self.name = name

    def base_file_name(self):
        return self.name
