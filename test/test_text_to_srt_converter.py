import unittest

import data.register_subclasses
from img_to_text.dummy_converter import DummyConverter
from text_to_srt.text_to_srt_converter import TextToSRTConverter
from video_to_img.example_image_provider import ExampleImageProvider


class TestTextToSRTConverter(unittest.TestCase):

    def test_converter(self):
        subject = TextToSRTConverter(ExampleImageProvider(), DummyConverter())
