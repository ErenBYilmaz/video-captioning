import unittest
from unittest import skipIf

from img_to_text.minigpt4 import MiniGPT4Captioning
from img_to_text.minigpt4_assisted_gpt3 import MiniGPT4AssistedGPT3
from lib.util import port_open
from video_to_img.example_image_provider import ExampleImageProvider


@skipIf(not port_open('127.0.0.1', 7860), 'MiniGPT4 server not available')
class TestMiniGPT4AssistedGPT3(unittest.TestCase):
    def test_conversion(self):
        subject = MiniGPT4AssistedGPT3()
        img_data = ExampleImageProvider().image_metadata_list()[0]
        result = subject._convert(img_data)
        assert isinstance(result, str)
        assert len(result) > 0
        print(result)
