import os.path
import unittest

import data.register_subclasses
from img_to_text.dummy_converter import DummyConverter
from text_to_srt.text_to_srt_converter import TextToSRTConverter
from video_to_img.example_image_provider import ExampleImageProvider
from video_to_img.example_video_converter import ExampleVideoToImageConverter


class TestTextToSRTConverter(unittest.TestCase):

    def test_converter(self):
        subject = TextToSRTConverter(ExampleVideoToImageConverter(overwrite=False), DummyConverter())
        subject.create_srt()
        assert os.path.isfile(subject.srt_path())