import os

import pysrt
from PIL import Image, ImageFont, ImageDraw
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from resources import resources
from text_to_srt.text_to_srt_converter import SRTCreator


class TextOverlayCreator:
    def __init__(self, srt_converter: SRTCreator):
        self.srt_converter = srt_converter

    def create_video_with_overlay(self):
        self.srt_converter.create_srt()
        srt_path = self.srt_converter.srt_path()
        srt_data = pysrt.open(srt_path, encoding='utf-8')
        base_video = self.image_provider().video_path
        video = VideoFileClip(base_video)
        captions = []

        for srt_step in srt_data:
            caption = srt_step.text
            start = srt_step.start.ordinal / 1000
            duration = srt_step.duration.ordinal / 1000
            pil_img = self.overlay_image_for_video(caption, video)
            tmp_img_path = os.path.join(resources.img_dir_path(), "frame_with_text_from_pillow2.png")
            pil_img.save(tmp_img_path)
            caption = ImageClip(tmp_img_path).set_start(start).set_duration(duration).set_pos((0, 0))
            captions.append(caption)
        result = CompositeVideoClip([video, *captions])

        print('Writing video file to', self.output_video_path())
        result.write_videofile(self.output_video_path(), fps=video.fps)

    def overlay_image_for_video(self, caption, video):
        lines = caption.splitlines()
        pil_img = Image.new('RGBA', (1280, 664), (0, 0, 0, 0))
        image_editable = ImageDraw.Draw(pil_img)
        box_path = os.path.join(resources.resource_dir_path(), "Frame at 00-00-16_overlay-black-background.png")
        foreground = Image.open(box_path)
        pil_img.paste(foreground, (0, 0), foreground)
        offset = 15
        if len(lines) > 0:
            text = self.get_wrapped_text(lines[0], font=self.font_1(), line_length=380)
            image_editable.text((80, offset), text, (237, 230, 211), font=(self.font_1()))
            offset += 15 + self.get_text_dimensions(text, self.font_1())[1]
        if len(lines) > 1:
            text = self.get_wrapped_text(lines[1], font=self.font_2(), line_length=320)
            image_editable.text((140, offset), text, (237, 230, 211), font=(self.font_2()))
            offset += 15 + self.get_text_dimensions(text, self.font_2())[1]
        if len(lines) > 2:
            text = '\n'.join(lines[2:])
            text = self.get_wrapped_text(text, font=self.font_3(), line_length=380)
            image_editable.text((80, offset), text, (237, 230, 211), font=(self.font_3()))
            offset += 15 + self.get_text_dimensions(text, self.font_3())[1]
        pil_img.resize((video.w, video.h), resample=Image.BICUBIC)
        return pil_img

    def get_wrapped_text(self, text: str,
                         font: ImageFont.FreeTypeFont,
                         line_length: int):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if font.getlength(line) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)

    def get_text_dimensions(self, text_string, font):
        # https://stackoverflow.com/a/46220683/9263761
        ascent, descent = font.getmetrics()

        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent
        num_lines = len(text_string.splitlines())
        text_height *= num_lines

        return (text_width, text_height)

    def font_1(self):
        return ImageFont.truetype(os.path.join(resources.resource_dir_path(), 'Arial Black.ttf'), 35)

    def font_2(self):
        return ImageFont.truetype(os.path.join(resources.resource_dir_path(), 'Arial Italic.ttf'), 24)

    def font_3(self):
        return ImageFont.truetype(os.path.join(resources.resource_dir_path(), 'Arial.ttf'), 20)

    def caption_duration(self):
        return self.srt_converter.caption_duration

    def caption_offset(self):
        return self.srt_converter.caption_offset

    def image_provider(self):
        return self.srt_converter.image_provider

    def output_video_path(self):
        base_path = os.path.dirname(self.image_provider().video_path)
        srt_file_path = os.path.join(base_path, f"{os.path.basename(self.image_provider().video_file_name()).split('.')[0]}_overlay.mp4")
        return srt_file_path
