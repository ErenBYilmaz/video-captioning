from typing import List

from video_to_img.image_data_provider import ImageDataProvider


class VideoToImageConverter(ImageDataProvider):
    def __init__(self, video_path: str):
        self.video_path = video_path

    def image_paths(self) -> List[str]:
        raise NotImplementedError('TO DO extract images from video')
