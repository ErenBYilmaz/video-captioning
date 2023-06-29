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
        pil_img = Image.new('RGBA', (video.w, video.h), (0, 0, 0, 0))
        image_editable = ImageDraw.Draw(pil_img)
        box_path = os.path.join(resources.resource_dir_path(), "Frame at 00-00-16_overlay-black-background.png")
        foreground = Image.open(box_path)
        pil_img.paste(foreground, (0, 0), foreground)
        if len(lines) > 0:
            image_editable.text((80, 15), lines[0], (237, 230, 211), font=(self.font_1()))
        if len(lines) > 1:
            image_editable.text((140, 80), lines[1], (237, 230, 211), font=(self.font_2()))
        if len(lines) > 2:
            image_editable.text((80, 155), '\n'.join(lines[2]), (237, 230, 211), font=(self.font_3()))
        return pil_img

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
