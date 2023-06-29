from data.image_metadata import ImageMetadata
from img_to_text.image_to_text_converter import ImageToCaptionConverter
from img_to_text.minigpt4 import MiniGPT4Captioning
from img_to_text.minigpt4_hugging_face import MiniGPT4HuggingFaceInterface
from img_to_text.openai_interface import OpenAIInterface


class MiniGPT4AssistedGPT3(ImageToCaptionConverter):
    def __init__(self):
        self.mini_gpt4 = MiniGPT4Captioning(self.default_minigpt4_prompt())
        self.interface = OpenAIInterface()

    def default_prompt(self, description: str):
        return f'''
        You are part of an image processing pipeline. You will be given an automatically generated description of an image that is part of a video.
        The goal of the pipeline is to derive some interesting information about that image and to display it for some seconds on the screen as an overlay on top of the video.
        Your role in this pipeline is to summarize the information in the description and to provide a caption for the video.
        Here is the description:
        ```
        {description}
        ```
        '''

    def default_minigpt4_prompt(self):
        return '''
        You are part of an image processing pipeline.
        You are given an an image that is part of a video.
        The goal of the pipeline is to derive some interesting information about that image and to display it for some seconds on the screen as an overlay on top of the video.
        Your role in this pipeline is to summarize what is displayed in the image including some informative and interesting facts.
        Your summary will then be used by a later pipeline component to create a caption for the video.
        '''

    def _convert(self, img_data: ImageMetadata) -> str:
        minigpt4_caption = self.mini_gpt4.cached_convert(img_data)
        prompt = self.default_prompt(minigpt4_caption)
        gpt_caption = self.interface.send_prompt(prompt)
        return gpt_caption
