import datetime
import os
from typing import List

import cv2

from data.image_metadata_from_video import ImageMetadataFromVideo
from resources import resources
from resources.resources import img_dir_path
from video_to_img.image_data_provider import ImageDataProvider


class VideoToImageConverter(ImageDataProvider):
    def __init__(self, video_path: str, overwrite=False):
        self.video_path = video_path
        self.overwrite = overwrite
        assert os.path.isfile(video_path)

    def extract_image_at_timestamp(self, timestamp: int) -> str:
        video_file = os.path.join(resources.resource_dir_path(), 'VIDEO_CAPTIONING_DEMO_VIDEO.mp4')
        video = cv2.VideoCapture(video_file)

        nr_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)

        frame_nr = timestamp * fps
        if nr_frames < frame_nr:
            raise ValueError(f'Frame at second {timestamp} is not in the video (frame {nr_frames}/{nr_frames})')

        video.set(1, frame_nr)
        success, frame = video.read()
        metadata = ImageMetadataFromVideo(image_type='jpg',
                                          timestamp=timestamp,
                                          video_path=self.video_path)
        to_path = metadata.image_path()
        if not self.overwrite:
            assert not os.path.isfile(to_path)
        metadata.save_to_disk()
        print('Writing to', to_path)
        success = cv2.imwrite(to_path, frame)
        assert success
        assert os.path.isfile(to_path)
        return to_path

    def image_paths(self) -> List[str]:
        timestamp = 0
        images = []
        while True:
            try:
                img_path = self.extract_image_at_timestamp(timestamp=timestamp)
            except ValueError as e:
                if str(e).startswith('Frame') and 'not in the video' in str(e):
                    break
            else:
                images.append(img_path)
            timestamp += 10
        return images
