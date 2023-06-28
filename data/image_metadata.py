import json
import os
from abc import ABCMeta
from typing import Dict, Any

import data.register_subclasses
from lib.util import EBC


class ImageMetadata(EBC, metaclass=ABCMeta):
    def __init__(self, image_type):
        self.image_type = image_type
        self._base_path = 'img'

    def image_filename(self):
        return self.base_file_name() + '.' + self.image_type

    def json_filename(self):
        return self.base_file_name() + '.json'

    def json_path(self):
        return os.path.join(self.base_path(), self.json_filename())

    def image_path(self):
        return os.path.join(self.base_path(), self.image_filename())

    def base_path(self):
        return self._base_path

    def base_file_name(self):
        raise NotImplementedError('Abstract method')

    def save_to_disk(self):
        with open(self.json_path(), 'w') as f:
            json.dump(self.to_json(), f, indent=4)

    @classmethod
    def from_json_file(cls, path: str):
        data.register_subclasses.register_subclasses()
        result = super().from_json_file(path)
        result._base_path = os.path.dirname(path)
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        del result['_base_path']
        return result

    def identifier(self):
        return self.base_file_name()


