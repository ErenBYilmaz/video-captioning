from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter
from lib.util import EBC


class DummyConverter(ImageToCaptionConverter):
    def _convert(self, img_data: ImageMetadata) -> str:
        return "'Fancy caption'"
