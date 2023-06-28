import datetime
import os.path

from data.image_metadata import ImageMetadata


class ImageMetadataFromVideo(ImageMetadata):
    def __init__(self, image_type: str, timestamp: float, video_path: str):
        super().__init__(image_type=image_type)
        self.timestamp = timestamp
        self.video_path = video_path

    def base_file_name(self):
        """
        :return: The base file name is composed of the provider name and the timestamp.
        the timestamp is formatted as follows: HH-MM-SS-MS
        """
        delta = datetime.timedelta(seconds=self.timestamp)
        hours = delta.total_seconds() // 3600
        minutes = (delta.total_seconds() - hours * 3600) // 60
        ms = delta.microseconds // 1000
        seconds = delta.total_seconds() - hours * 3600 - minutes * 60 - ms / 1000
        ts_str = f'{int(hours):02d}-{int(minutes):02d}-{int(seconds):02d}-{int(ms):03d}'
        return f'{type(self).__name__}_{os.path.basename(self.video_path)}_{ts_str}'.replace('.', '_')
