import os.path
import unittest

import data.register_subclasses
from img_to_text.dummy_converter import DummyConverter
from text_overlay.text_overlay import TextOverlayCreator
from text_to_srt.text_to_srt_converter import SRTCreator
from video_to_img.example_image_provider import ExampleImageProvider
from video_to_img.example_video_converter import ExampleVideoToImageConverter


class TestTextOverlay(unittest.TestCase):

    def test_creator(self):
        subject = TextOverlayCreator(SRTCreator(ExampleVideoToImageConverter(overwrite=False), DummyConverter()))
        subject.create_video_with_overlay()
        assert os.path.isfile(subject.output_video_path())