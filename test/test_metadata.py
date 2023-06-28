import os
import unittest

from data.image_metadata import ImageMetadata
from data.image_metadata_from_video import ImageMetadataFromVideo
from data.named_image_metadata import NamedImageMetadata
from resources.resources import resource_dir_path
from video_to_img.example_image_provider import ExampleImageProvider
from video_to_img.single_image_provider import SingleImageProvider


class TestImageMetadata(unittest.TestCase):
    def test_json_conversion(self):
        provider = ExampleImageProvider()
        img_metadata = provider.image_metadata_list()[0]
        assert isinstance(img_metadata, ImageMetadata)
        self.assertEqual(img_metadata.to_json(), ImageMetadata.from_json(img_metadata.to_json()).to_json())

    def test_file_name_formatting(self):
        provider = ExampleImageProvider()
        img_metadata = provider.image_metadata_list()[0]
        assert isinstance(img_metadata, NamedImageMetadata)
        assert os.path.isfile(img_metadata.image_path()), img_metadata.image_path()
        assert os.path.isfile(img_metadata.json_path())


class TestImageMetadataFromVideo(unittest.TestCase):
    def test_json_conversion(self):
        provider = SingleImageProvider(os.path.join(resource_dir_path(), 'ImageMetadataFromVideo_ExampleVideo_mp4_00090_00.jpg'))
        img_metadata = provider.image_metadata_list()[0]
        assert isinstance(img_metadata, ImageMetadataFromVideo)
        self.assertEqual(img_metadata.to_json(), ImageMetadata.from_json(img_metadata.to_json()).to_json())

    def test_file_name_formatting(self):
        provider = SingleImageProvider(os.path.join(resource_dir_path(), 'ImageMetadataFromVideo_ExampleVideo_mp4_00090_00.jpg'))
        img_metadata = provider.image_metadata_list()[0]
        assert os.path.isfile(img_metadata.image_path()), img_metadata.image_path()
        assert os.path.isfile(img_metadata.json_path())
