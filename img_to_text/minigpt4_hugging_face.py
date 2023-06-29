import json
import uuid
from typing import Optional, Dict

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
        requests.get(self.base_url()) #, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'})
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

    def receive_answer(token):
        """Waits until the server sends an answer that contains the desired request_token.
        All intermediate requests are also collected for later use, or, if they contain no token, they are just printed out.
        """
        if token in response_collection:
            json_content = response_collection[token]
            del response_collection[token]
            return json_content

        json_content = {}
        while 'request_token' not in json_content or json_content['request_token'] != token:
            if 'request_token' in json_content:
                response_collection[json_content['request_token']] = json_content
            received = server.do_some_requests.current_websocket.current_websocket.recv_data_frame()[1].data
            content = received.decode('latin-1')
            # formatted_content = re.sub(r'{([^}]*?):(.*?)}', r'\n{\g<1>:\g<2>}', content)
            json_content = json.loads(content)
            formatted_content = compact_dict_string(json_content, max_line_length=300)
            print('Received through websocket: ' + formatted_content)

        return json_content

    step_idx = 0

    def websocket_request(route: str, data: Dict, allow_failure=False) -> Dict:
        token = send_websocket_request_without_waiting_for_answer(route, data)
        json_content = receive_answer(token)
        print()

        status_code = json_content['http_status_code']
        if status_code == 200:
            pass
        elif status_code == 451:  # Copyright problems, likely a bug in the upload filter
            # Try again
            return websocket_request(route, data)
        else:
            if EXIT_ON_FAILED_REQUEST and (not allow_failure or status_code >= 500):
                if not server.do_some_requests.current_websocket.current_websocket.connected:
                    server.do_some_requests.current_websocket.current_websocket.close()
                sys.exit(status_code)
            if not allow_failure:
                failed_requests.append((route, status_code))

        successful_requests.append(
            SuccessfulRequest(
                route=route,
                data=data,
                response=json_content['body'],
                status_code=status_code,
            )
        )
        return json_content['body']

    def send_websocket_request_without_waiting_for_answer(route: str, data: Dict):
        if not server.do_some_requests.current_websocket.current_websocket.connected:
            ws_host = HOST.replace('https://', 'wss://').replace('http://', 'ws://')
            websocket.WebSocket().connect(ws_host + '/websocket')
            server.do_some_requests.current_websocket.current_websocket.connect(ws_host + '/websocket')
        token = str(uuid4())
        json_data = {'route': route, 'body': data, 'request_token': token}
        data_to_send = encode_json(json_data)
        global step_idx
        # print(f'Sending to websocket (Step {step_idx}): {data_to_send[:200]}')
        print(f'Sending to websocket (Step {step_idx}): {compact_dict_string(json_data, max_line_length=300)}')
        step_idx += 1
        # print('Sending to websocket:', str(data.decode('latin-1')).replace('{', '\n{')[1:])
        server.do_some_requests.current_websocket.current_websocket.send(data_to_send, opcode=2)
        return token

    def encode_json(self, json_data):
        return json.dumps(json_data)
