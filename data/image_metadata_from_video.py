import os.path
from typing import Literal

from data.image_metadata import ImageMetadata


class ImageMetadataFromVideo(ImageMetadata):
    def __init__(self, image_type: Literal['png', 'jpg'], timestamp: float, video_path: str):
        super().__init__(image_type=image_type)
        self.timestamp = timestamp
        self.video_path = video_path

    def base_file_name(self):
        """
        :return: The base file name is composed of the provider name and the timestamp,
        where the timestamp is padded by leading zeros to 5 digits and also includes 2 decimals.
        """
        return f'{type(self).__name__}_{os.path.basename(self.video_path)}_{self.timestamp:08.2f}'.replace('.', '_')
