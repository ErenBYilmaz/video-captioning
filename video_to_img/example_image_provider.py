import os

from resources.resources import resource_dir_path
from video_to_img.single_image_provider import SingleImageProvider


class ExampleImageProvider(SingleImageProvider):
    def __init__(self):
        super().__init__(os.path.join(resource_dir_path(), 'example.jpg'))
