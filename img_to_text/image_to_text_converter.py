from data.image_metadata import ImageMetadata
from lib.util import EBC


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