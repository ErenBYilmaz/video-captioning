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

    def json_path(self) -> List[str]:
        p = self.image_path()
        fn, ext = os.path.splitext(p)
        return fn + '.json'
