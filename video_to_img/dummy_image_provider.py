from typing import List

from video_to_img.image_data_provider import ImageDataProvider
from video_to_img.single_image_provider import SingleImageProvider


class DummyImageProvider(SingleImageProvider):
    def __init__(self):
        super().__init__('resources/example.jpg')
