import os.path
from typing import List

from data.image_metadata import ImageMetadataFromVideo


class ImageDataProvider:
    def name(self):
        return type(self).__name__

    def image_paths(self) -> List[str]:
        """
        :return the path to the image file that should be processed.
        """
        raise NotImplementedError('Abstract method')

    def json_paths(self) -> List[str]:
        return [os.path.splitext(p)[0] + '.json' for p in self.image_paths()]

    def image_metadata_list(self) -> List[ImageMetadataFromVideo]:
        return [ImageMetadataFromVideo.from_json_file(p) for p in self.json_paths()]