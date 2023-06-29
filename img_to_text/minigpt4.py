from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter
from img_to_text.minigpt4_hugging_face import MiniGPT4HuggingFaceInterface


class MiniGPT4Captioning(ImageToCaptionConverter):
    def __init__(self):
        self.interface = MiniGPT4HuggingFaceInterface()

    def _convert(self, img_data: ImageMetadata) -> str:
        self.interface.upload_img(img_data)
        return self.interface.send_prompt(self.prompt())

    def prompt(self):
        return 'The image is a snapshot of a video file. Your job to create a caption for the video that should be informative and show some interesting fact(s). Please provide five words at most.'

