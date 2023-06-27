import os.path
from typing import List


class ImageDataProvider:
    def name(self):
        return type(self).__name__

    def image_paths(self) -> List[str]:
        """
        :return the path to the image file that should be processed.
        """
        raise NotImplementedError('Abstract method')

    def json_paths(self) -> List[str]:
        return [os.path.splitext(p)[0] + '.json' for p in self.image_paths()]
