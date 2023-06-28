import unittest

import data.register_subclasses
from img_to_text.dummy_converter import DummyConverter
from video_to_img.example_image_provider import ExampleImageProvider


class TestDummyConverter(unittest.TestCase):

    def test_converter(self):
        subject = DummyConverter()
        img_data = ExampleImageProvider().image_metadata_list()[0]
        result = subject.convert(img_data)
        assert isinstance(result, str)
        assert len(result) > 0
