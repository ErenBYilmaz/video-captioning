import unittest

from img_to_text.minigpt4 import MiniGPT4Captioning
from video_to_img.example_image_provider import ExampleImageProvider


class TestMiniGPT4(unittest.TestCase):
    def test_mini_gpt4(self):
        subject = MiniGPT4Captioning()
        img_data = ExampleImageProvider().image_metadata_list()[0]
        result = subject.convert(img_data)
        assert isinstance(result, str)
        assert len(result) > 0
