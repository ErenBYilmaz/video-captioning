import datetime
import json
import os
from typing import Literal


class ImageMetadata:
    def __init__(self, created_by_provider: str, dt_created: float, image_type: Literal['png', 'jpg']):
        self.dt_created = dt_created
        self.created_by_provider = created_by_provider
        self.image_type = image_type

    def image_filename(self):
        return self._base_name() + '.' + self.image_type

    def json_filename(self):
        return self._base_name() + '.json'

    def json_path(self):
        return os.path.join('img', self.json_filename())

    def _base_name(self):
        return f'{self.created_by_provider}_{self.dt_str()}'

    def dt_str(self) -> str:
        return datetime.datetime.fromtimestamp(self.dt_created).strftime('%Y-%m-%d_%H-%M-%S')

    def save_to_disk(self):
        with open(self.json_path(), 'w') as f:
            json.dump(self.to_json(), f, indent=4)
