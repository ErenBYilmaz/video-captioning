import os
import openai

from data.image_metadata import ImageMetadata

class OpenAIInterface:
    def send_prompt(self, prompt: str):
        print()
        print(f'Sending to chatgpt: {prompt}')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        message = [{"role": "user",
                    "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.2,
            max_tokens=1000,
            frequency_penalty=0.0
        )
        result = response['choices'][0]['message']['content']
        print(f'Received from chatgpt: {result}')
        return result

    @staticmethod
    def list_models():
        models = openai.Model.list()
        print([model['id'] for model in models['data']])