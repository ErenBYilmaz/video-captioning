import os
import unittest

from video_to_img.example_video_converter import ExampleVideoToImageConverter


class TestExampleVideoToImageConverter(unittest.TestCase):
    def test_video_to_image_converter(self):
        subject = ExampleVideoToImageConverter()
        assert len(subject.image_metadata_list()) > 1
        assert len(subject.image_metadata_list()) == len(subject.image_paths()) == len(subject.json_paths())
        for metadata in subject.image_metadata_list():
            assert os.path.isfile(metadata.image_path())
            assert os.path.isfile(metadata.json_path())
