from text_overlay.text_overlay import TextOverlayCreator
from video_to_img.example_video_converter import ExampleVideoToImageConverter
from video_to_img.youtube_to_img import YoutubeToImageConverter

from img_to_text.minigpt4_assisted_gpt3 import MiniGPT4AssistedGPT3
from text_to_srt.text_to_srt_converter import SRTCreator


def main():
    # MiniGPT4AssistedGPT3().clear_output_cache()
    # pipeline = SRTCreator(YoutubeToImageConverter('https://www.youtube.com/watch?v=uQITWbAaDx0'), MiniGPT4AssistedGPT3())
    # pipeline = SRTCreator(ExampleVideoToImageConverter(), MiniGPT4AssistedGPT3())
    for url in [
        'https://www.youtube.com/watch?v=STPvOxUDekU',
        'https://www.youtube.com/watch?v=wPYx-kRiXxA',
        'https://www.youtube.com/shorts/COVbxW9h58E',
        'https://www.youtube.com/watch?v=IUN664s7N-c',
        'https://www.youtube.com/watch?v=3Oc7cgIcEFg&t=56s',
    ]:
        pipeline = SRTCreator(YoutubeToImageConverter(url), MiniGPT4AssistedGPT3())
        pipeline = TextOverlayCreator(pipeline)
        pipeline.create_video_with_overlay()


if __name__ == '__main__':
    main()
