import json
import uuid
from typing import Optional, Dict

import cachetools
import requests
import websocket

from data.image_metadata import ImageMetadata

class MiniGPT4HuggingFaceInterface:
    def __init__(self):
        self.uploaded_img = False
        self.socket: Optional[websocket.WebSocket] = None
        self.session_id: Optional[str] = None

    def base_url(self):
        return 'http://host.docker.internal:7860'
        # return 'ws://127.0.0.1:7860'
        # return 'wss://damo-nlp-sg-video-llama.hf.space'

    def upload_img(self, img_data: ImageMetadata):
        requests.get(self.base_url())
        self.session_id = str(uuid.uuid4())

        json_data = {
            'fn_index': 0,
            'event_data': None,
            'session_hash': self.session_id,
            'data': [f'data:image/{img_data.image_type};base64,' + img_data.base64().decode('ascii'), '', None]
        }
        response = requests.post(self.predict_url(), json=json_data, headers={'Content-Type': 'application/json'})
        data = response.json()
        print('received', data)

        self.uploaded_img = True

    def predict_url(self):
        return self.base_url() + '/run/predict'


    def send_prompt(self, prompt: str):
        if not self.uploaded_img:
            raise RuntimeError('Image not uploaded')
        json_data = {
            'fn_index': 2,
            'data': [[[prompt, None]], None, None, 1, 1],
            'event_data': None,
            'session_hash': self.session_id,
        }
        response = requests.post(self.predict_url(), json=json_data, headers={'Content-Type': 'application/json'})
        data = response.json()
        print('received', data)
        return data['data'][0][0][1]

