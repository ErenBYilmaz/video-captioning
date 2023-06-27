from typing import List

from video_to_img.image_data_provider import ImageDataProvider


class SingleImageProvider(ImageDataProvider):
    def __init__(self, img_path: str):
        self.img_path = img_path

    def image_paths(self) -> List[str]:
        return [self.img_path]
