from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter


class MiniGPT4Interface:
    def __init__(self):
        self.uploaded_img = False

    def upload_img(self, img_data: ImageMetadata):
        raise NotImplementedError('TO DO')
        self.uploaded_img = True

    def send_prompt(self, prompt: str):
        if not self.uploaded_img:
            raise RuntimeError('Image not uploaded')
        raise NotImplementedError('TO DO')


class MiniGPT4Captioning(ImageToCaptionConverter):
    def __init__(self):
        self.interface = MiniGPT4Interface()

    def convert(self, img_data: ImageMetadata) -> str:
        self.interface.upload_img(img_data)
        prompt = 'The image is a snapshot of a video file. You job to create a caption for the video that should be informative and show some interesting fact(s). Please provide five words at most.'
        return self.interface.send_prompt(prompt)
