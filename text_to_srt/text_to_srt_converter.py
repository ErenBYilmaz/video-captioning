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


class TextToSRTConverter:
    caption_duration = 5000  # default caption time in milliseconds
    caption_offset = -2000

    def __init__(self, image_provider: VideoToImageConverter, image_to_txt_converter: ImageToCaptionConverter):
        self.image_provider = image_provider
        self.image_to_txt_converter = image_to_txt_converter

    def create_srt(self):
        image_metadata = self.image_provider.image_metadata_list()
        srt_file_path = self.srt_path()
        # subs = pysrt.open()
        subs = pysrt.SubRipFile()

        for img_meta in image_metadata:
            caption = self.image_to_txt_converter.cached_convert(img_meta)
            if caption == '':
                continue
            timestamp = img_meta.timestamp
            subs.append(pysrt.SubRipItem(index=len(subs) + 1,
                                         text=caption,
                                         start=timestamp * 1000 + self.caption_offset,
                                         end=timestamp * 1000 + self.caption_offset + self.caption_duration))

        print('Writing srt file to', srt_file_path)
        subs.save(srt_file_path, encoding='utf-8')

    def srt_path(self):
        base_path = os.path.dirname(self.image_provider.video_path)
        srt_file_path = os.path.join(base_path, f"{os.path.basename(self.image_provider.video_file_name()).split('.')[0]}.srt")
        return srt_file_path
