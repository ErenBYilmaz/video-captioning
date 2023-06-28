import os.path

import cv2

from resources import resources
from resources.resources import img_dir_path

video_file = os.path.join(resources.resource_dir_path(), 'VIDEO_CAPTIONING_DEMO_VIDEO.mp4')
assert os.path.isfile(video_file)
video = cv2.VideoCapture(video_file)

nr_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

timestamp = '00:00:30'
# timestamp = input('Enter timestamp in hh:mm:ss format: ')

timestamp_list = timestamp.split(':')
hh, mm, ss = timestamp_list
timestamp_list_floats = [float(i) for i in timestamp_list]
hours, minutes, seconds = timestamp_list_floats

frame_nr = hours * 3600 * fps + minutes * 60 * fps + seconds * fps

video.set(1, frame_nr)
success, frame = video.read()
to_path = os.path.join(img_dir_path(), f'Frame-at-{hh}{mm}{ss}.jpg')
print('Writing to', to_path)
success = cv2.imwrite(to_path, frame)
assert success
assert os.path.isfile(to_path)