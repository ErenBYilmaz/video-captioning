from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter


class MiniGP4Interface:
    def __init__(self):
        self.uploaded_img = False

    def upload_img(self, img_data: ImageMetadata):
        raise NotImplementedError('TO DO')
        self.uploaded_img = True

    def send_text(self, text: str):
        if not self.uploaded_img:
            raise RuntimeError('Image not uploaded')
        raise NotImplementedError('TO DO')


class MiniGP4Captioning(ImageToCaptionConverter):
    def __init__(self):
        self.interface = MiniGP4Interface()

    def convert(self, img_data: ImageMetadata) -> str:
        self.interface.upload_img(img_data)
        return self.interface.send_text(
            'The image is a snapshot of a video file. You job to create a caption for the video that should be informative and show some interesting fact(s). Please provide five words at most.')
