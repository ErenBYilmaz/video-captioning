import os

from resources import resources
from video_to_img.video_to_image_converter import VideoToImageConverter


class YoutubeToImageConverter(VideoToImageConverter):
    def __init__(self, video_url: str, overwrite=False):
        self.video_url = video_url
        super().__init__(video_path=self._download_video(), overwrite=overwrite)

    def _download_video(self):
        # noinspection PyUnresolvedReferences
        import lib.monkey_patch_pytube
        import pytube
        yt = pytube.YouTube(self.video_url)
        video = yt.streams.filter(file_extension='mp4', progressive=True).first()
        to_dir = resources.img_dir_path()
        video.download(to_dir)
        return os.path.join(to_dir, video.default_filename)
