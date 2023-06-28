from data.image_metadata import ImageMetadata
from lib.util import EBC


class ImageToCaptionConverter(EBC):
    def convert(self, img_data: ImageMetadata) -> str:
        raise NotImplementedError('Abstract method')
