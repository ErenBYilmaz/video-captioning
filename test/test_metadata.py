import unittest

from data.image_metadata import ImageMetadata
from video_to_img.example_image_provider import ExampleImageProvider


class TestImageMetadata(unittest.TestCase):
    def test_json_conversion(self):
        provider = ExampleImageProvider()
        img_metadata = provider.image_metadata_list()[0]
        assert isinstance(img_metadata, ImageMetadata)
        self.assertEqual(img_metadata.to_json(), ImageMetadata.from_json(img_metadata.to_json()).to_json())
        assert img_metadata.created_by_provider == provider.name()
