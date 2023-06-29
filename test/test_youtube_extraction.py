import os
import unittest

from video_to_img.youtube_to_img import YoutubeToImageConverter


class TestExampleVideoToImageConverter(unittest.TestCase):
    def test_video_to_image_converter(self):
        subject = YoutubeToImageConverter('https://www.youtube.com/watch?v=uQITWbAaDx0')
        assert len(subject.image_metadata_list()) > 1
        assert len(subject.image_metadata_list()) == len(subject.image_paths()) == len(subject.json_paths())
        for metadata in subject.image_metadata_list():
            assert os.path.isfile(metadata.image_path())
            assert os.path.isfile(metadata.json_path())
