import os.path
import unittest

from video_to_img.example_image_provider import ExampleImageProvider


class TestExampleImageProvider(unittest.TestCase):
    def test_get_image(self):
        provider = ExampleImageProvider()
        for img_path in provider.image_paths():
            assert os.path.isfile(img_path)
        for json_path in provider.json_paths():
            assert os.path.isfile(json_path)
