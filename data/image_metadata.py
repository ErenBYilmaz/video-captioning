import datetime
import json
import os
from abc import ABCMeta, abstractmethod
from typing import Literal

from lib.util import EBC


class ImageMetadata(EBC, metaclass=ABCMeta):
    def __init__(self, image_type, created_by_provider):
        self.image_type = image_type
        self.created_by_provider = created_by_provider

    def image_filename(self):
        return self.base_file_name() + '.' + self.image_type

    def json_filename(self):
        return self.base_file_name() + '.json'

    def json_path(self):
        return os.path.join('img', self.json_filename())

    def image_path(self):
        return os.path.join('img', self.image_filename())

    def base_file_name(self):
        raise NotImplementedError('Abstract method')

    def save_to_disk(self):
        with open(self.json_path(), 'w') as f:
            json.dump(self.to_json(), f, indent=4)

    def identifier(self):
        return self.base_file_name()


class ImageMetadataFromVideo(ImageMetadata):
    def __init__(self, created_by_provider: str, image_type: Literal['png', 'jpg'], timestamp: float, video_path: str):
        super().__init__(created_by_provider=created_by_provider, image_type=image_type)
        self.timestamp = timestamp
        self.video_path = video_path

    def base_file_name(self):
        """
        :return: The base file name is composed of the provider name and the timestamp,
        where the timestamp is padded by leading zeros to 5 digits and also includes 2 decimals.
        """
        return f'{self.created_by_provider}_{self.timestamp:08.2f}'
