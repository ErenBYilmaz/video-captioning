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

    def base_file_name(self):
        raise NotImplementedError('Abstract method')

    def save_to_disk(self):
        with open(self.json_path(), 'w') as f:
            json.dump(self.to_json(), f, indent=4)


class ImageMetadataFromVideo(ImageMetadata):
    def __init__(self, created_by_provider: str, dt_created: float, image_type: Literal['png', 'jpg']):
        super().__init__(created_by_provider=created_by_provider, image_type=image_type)
        self.dt_created = dt_created

    def dt_str(self) -> str:
        return datetime.datetime.fromtimestamp(self.dt_created).strftime('%Y-%m-%d_%H-%M-%S')

    def base_file_name(self):
        return f'{self.created_by_provider}_{self.dt_str()}'