from img_to_text.minigpt4 import MiniGPT4Captioning
from text_to_srt.text_to_srt_converter import TextToSRTConverter
from video_to_img.example_video_converter import ExampleVideoToImageConverter


def main():
    pipeline = TextToSRTConverter(ExampleVideoToImageConverter(), MiniGPT4Captioning())
    pipeline.create_srt()


if __name__ == '__main__':
    main()
