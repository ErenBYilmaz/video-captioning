
from data.image_metadata import ImageMetadata


class NamedImageMetadata(ImageMetadata):
    def __init__(self, image_type: str, name: str):
        super().__init__(image_type=image_type)
        self.name = name

    def base_file_name(self):
        return self.name
