import os
import unittest
import pysrt

import data.register_subclasses
from img_to_text.dummy_converter import DummyConverter
from img_to_text.image_to_text_converter import ImageToCaptionConverter
from video_to_img.example_image_provider import ExampleImageProvider
from video_to_img.example_video_converter import ExampleVideoToImageConverter
from video_to_img.image_data_provider import ImageDataProvider
from video_to_img.video_to_image_converter import VideoToImageConverter


class TextToSRTConverter():
    def __init__(self, image_provider: VideoToImageConverter, image_to_txt_converter: ImageToCaptionConverter):
        self.image_provider = image_provider
        self.image_to_txt_converter = image_to_txt_converter

    def create_srt(self):
        image_metadata = self.image_provider.image_metadata_list()
        base_path = os.path.dirname(self.image_provider.video_path)
        subs = pysrt.open(os.path.join(base_path, f"{self.image_provider.video_file_name()}.srt"))
        for img_meta in image_metadata:
            caption = self.image_to_txt_converter.convert(img_meta)
