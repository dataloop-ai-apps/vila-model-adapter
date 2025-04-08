import base64
import dtlpy as dl
import logging
import os
import requests
import json
import time
import subprocess
import threading
from io import BytesIO
from PIL import Image

logger = logging.getLogger('vila-adapter')


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity):
        super().__init__(model_entity)
        self.vila_server_process = None
        self.vila_base_url = None

    def load(self, local_path, **kwargs):
        """Load VILA model server and configuration"""
        self.adapter_defaults.upload_annotations = False

        # Check if VILA model path is provided
        model_path = self.configuration.get("model_path", "Efficient-Large-Model/NVILA-15B")
        conv_mode = self.configuration.get("conv_mode", "auto")
        port = self.configuration.get("port", 8000)
        self.vila_base_url = f"http://localhost:{port}"

        # Start VILA server as a subprocess
        server_command = [
            "python",
            "-W",
            "ignore",
            "/app/server.py",
            "--model-path",
            model_path,
            "--conv-mode",
            conv_mode,
            "--port",
            str(port),
        ]

        self.vila_server_process = subprocess.Popen(
            server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        # Start threads to monitor stdout and stderr
        threading.Thread(
            target=self._log_stream, args=(self.vila_server_process.stdout, logger.info), daemon=True
        ).start()
        threading.Thread(
            target=self._log_stream, args=(self.vila_server_process.stderr, logger.error), daemon=True
        ).start()

        # Wait for server to start
        logger.info(f"Starting VILA server with model: {model_path}")
        time.sleep(10)  # Give some time for the server to start

        # Test connection to the server
        try:
            response = requests.get(f"{self.vila_base_url}")
            if response.status_code == 404:
                logger.info("VILA server is running")
            else:
                logger.error(f"Unexpected response from VILA server: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to VILA server: {e}")
            raise ValueError(f"Failed to start VILA server: {e}")

    def _log_stream(self, stream, log_func):
        """Helper function to log output from subprocesses"""
        for line in stream:
            log_func(line.strip())

    def call_model(self, messages):
        """Call the VILA model server with messages"""
        stream = self.configuration.get("stream", True)
        max_tokens = self.configuration.get("max_tokens", 512)
        temperature = self.configuration.get("temperature", 0.2)
        top_p = self.configuration.get("top_p", 0.9)
        model_name = self.configuration.get("model_path", "Efficient-Large-Model/NVILA-15B").split("/")[-1]

        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        try:
            if stream:
                response = requests.post(
                    f"{self.vila_base_url}/chat/completions", headers=headers, json=data, stream=True
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                yield content
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {line}")
            else:
                response = requests.post(f"{self.vila_base_url}/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

                if isinstance(content, list):
                    # Handle case where content is a list of text parts
                    for item in content:
                        if item.get("type") == "text":
                            yield item.get("text", "")
                else:
                    yield content

        except Exception as e:
            logger.error(f"Error calling VILA model: {e}")
            yield f"Error: {str(e)}"

    def prepare_item_func(self, item: dl.Item):
        """Prepare item for VILA model"""
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def _get_image_base64(self, image_path):
        """Convert image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def predict(self, batch, **kwargs):
        """Predict with VILA model"""
        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name

        for prompt_item in batch:
            # Get all messages including model annotations
            messages = []

            # Add system message
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Process all prompts
            for prompt in prompt_item.prompts:
                role = "user" if prompt.user_id else "assistant"

                # Handle text-only messages
                if all(msg.get('mimetype') == dl.PromptType.TEXT for msg in prompt.message.get('content', [])):
                    text_content = " ".join(
                        msg.get('value', '')
                        for msg in prompt.message.get('content', [])
                        if msg.get('mimetype') == dl.PromptType.TEXT
                    )
                    messages.append({"role": role, "content": text_content})
                else:
                    # Handle multimodal messages (text and images)
                    content_list = []

                    for msg in prompt.message.get('content', []):
                        if msg.get('mimetype') == dl.PromptType.TEXT:
                            content_list.append({"type": "text", "text": msg.get('value', '')})
                        elif msg.get('mimetype') in [dl.PromptType.IMAGE, 'image/jpeg', 'image/png']:
                            # For files in Dataloop, download them first
                            if isinstance(msg.get('value'), str) and msg.get('value').startswith('http'):
                                image_url = msg.get('value')
                            else:
                                # Download the image and convert to base64
                                temp_path = prompt_item.download(msg.get('value'), save_locally=True)
                                image_data = self._get_image_base64(temp_path)
                                image_url = f"data:image/jpeg;base64,{image_data}"

                            content_list.append({"type": "image_url", "image_url": {"url": image_url}})

                    messages.append({"role": role, "content": content_list})

            # Add context from nearest items if available
            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items, add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            # Call the model and process the response
            stream_response = self.call_model(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(
                    message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response}]},
                    model_info={'name': model_name, 'confidence': 1.0, 'model_id': self.model_entity.id},
                )

        return []

    def __del__(self):
        """Clean up resources when the adapter is destroyed"""
        if self.vila_server_process:
            logger.info("Shutting down VILA server")
            self.vila_server_process.terminate()
            try:
                self.vila_server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vila_server_process.kill()


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="")
    item = dl.items.get(item_id="")
    a = ModelAdapter(model)
    a.predict_items([item])
