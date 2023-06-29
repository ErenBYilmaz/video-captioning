import json
import os

from data.image_metadata import ImageMetadata
from lib.util import EBC
from resources.resources import img_dir_path


class ImageToCaptionConverter(EBC):
    def _convert(self, img_data: ImageMetadata) -> str:
        raise NotImplementedError('Abstract method')

    def cached_convert(self, img_data: ImageMetadata) -> str:
        output = img_data.tool_outputs.get(self.name(), None)
        if output is None:
            self._convert_and_update_metadata(img_data)
            img_data.reload()
            assert self.name() in img_data.tool_outputs
        return output

    def _convert_and_update_metadata(self, img_data: ImageMetadata):
        print(self.name(), f': Computing caption for {img_data.image_path()}...')
        result = self._convert(img_data)
        img_data.tool_outputs[self.name()] = result
        img_data.save_to_disk()

    def name(self):
        return self.__class__.__name__

    def clear_output_cache(self):
        for filename in os.listdir(img_dir_path()):
            with open(os.path.join(img_dir_path(), filename), 'r') as f:
                img_data = ImageMetadata.from_json(json.load(f))
            if self.name() in img_data.tool_outputs:
                del img_data.tool_outputs[self.name()]
                img_data.save_to_disk()