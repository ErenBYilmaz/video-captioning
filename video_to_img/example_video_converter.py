import os

from resources import resources
from video_to_img.video_to_image_converter import VideoToImageConverter


class ExampleVideoToImageConverter(VideoToImageConverter):
    def __init__(self):
        super().__init__(video_path=os.path.join(resources.resource_dir_path(), 'VIDEO_CAPTIONING_DEMO_VIDEO.mp4'),
                         overwrite=True)
