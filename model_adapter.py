import base64
import dtlpy as dl
import logging
import requests
import time
import subprocess
from openai import OpenAI
import re

logger = logging.getLogger("vila-adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity):
        super().__init__(model_entity)
        self.vila_server_process = None
        self.vila_base_url = None
        self.client = None

    def load(self, local_path, **kwargs):
        """Load VILA model server and configuration"""
        self.adapter_defaults.upload_annotations = False

        # Check if VILA model path is provided
        model_path = self.configuration.get("model_path", "Efficient-Large-Model/NVILA-8B")
        conv_mode = self.configuration.get("conv_mode", "auto")
        port = self.configuration.get("port", 8000)
        self.vila_base_url = f"http://localhost:{port}"

        # Initialize OpenAI client that talks to the local VILA server. We use a fake
        # API‑key because the server does not validate it – it only expects the
        # header to exist in order to mimic the OpenAI API.
        self.client = OpenAI(
            base_url=self.vila_base_url, api_key=self.configuration.get("openai_api_key", "fake-key")
        )
        # Check if VILA server is already running
        server_running = False
        try:
            resp = requests.get(self.vila_base_url, timeout=2)
            if resp.status_code < 500:
                server_running = True
                logger.info(f"VILA server already running at {self.vila_base_url}, skipping start.")
        except requests.RequestException:
            pass

        # Start VILA server as a subprocess
        if not server_running:
            server_command = [
                "/opt/conda/envs/vila_env/bin/python",
                "-W",
                "ignore",
                "/tmp/app/VILA/custom_server.py",
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

            # Wait for server to start
            logger.info(f"Starting VILA server with model: {model_path}")
            self._wait_for_server_ready()

    def call_model(self, messages):
        """Call the VILA model server with messages"""
        stream = self.configuration.get("stream", False)
        max_tokens = self.configuration.get("max_tokens", 512)
        temperature = self.configuration.get("temperature", 0.2)
        top_p = self.configuration.get("top_p", 0.9)
        model_name = self.configuration.get("model_path", "Efficient-Large-Model/NVILA-15B").split("/")[-1]
        if not self.client:
            self.client = OpenAI(
                base_url=self.vila_base_url, api_key=self.configuration.get("openai_api_key", "fake-key")
            )

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                model=model_name,
            )

            if stream:
                for chunk in response:
                    # When streaming, the client returns `ChatCompletionChunk` objects
                    yield chunk.choices[0].delta.content or ""
            else:
                # Non‑streaming – a single response object is returned
                yield response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Error calling VILA model via OpenAI client: {e}")
            yield f"Error: {str(e)}"

    def prepare_item_func(self, item: dl.Item):
        """Prepare item for VILA model"""
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def _wait_for_server_ready(self, timeout: int = 60, interval: float = 1.0):
        """Poll the server root endpoint until it responds or the timeout is reached.

        Args:
            timeout: Maximum amount of seconds to wait.
            interval: Delay between consecutive probes in seconds.

        Raises:
            RuntimeError: If the server did not become responsive in the
                allotted time.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.vila_base_url, timeout=2)
                if response.status_code < 500:
                    logger.info("VILA server is running and responsive")
                    return
            except requests.RequestException:
                # The server is not up yet
                pass

            time.sleep(interval)

        raise RuntimeError(f"Timed out after {timeout}s while waiting for VILA server to start on {self.vila_base_url}")

    def _get_image_base64(self, image_path: str) -> str:
        """Read an image from *image_path* and return a base-64 encoded string."""
        with open(image_path, "rb") as img_fd:
            return base64.b64encode(img_fd.read()).decode("utf-8")

    def predict(self, batch, **kwargs):
        """Run inference on a batch of *PromptItem*s using the local VILA server.

        The method converts the Dataloop *PromptItem* structure to the message
        format expected by the OpenAI compatible endpoint exposed by the VILA
        server (see ``server.py``). It supports plain text, images and videos
        (either provided as URLs or embedded via Markdown such as
        ``[video_url](https://example.com/video.mp4)``).
        """
        # NOTE: System prompt is not allowed by model
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name
        video_frames = self.configuration.get("video_frames", 8)

        # Regex pattern to detect video markdown links – case insensitive.
        video_md_pattern = re.compile(r"\[video_url\]\(([^)]+)\)", re.IGNORECASE)

        for prompt_item in batch:
            prompt_item: dl.PromptItem = prompt_item
            # Generate messages using prompt_item.to_messages
            messages = prompt_item.to_messages(model_name=model_name)
            # Process messages to handle markdown video links within text
            processed_messages = []
            for message in messages:
                role = message['role']
                content = message['content']
                new_content_list = []

                # Content can be a string or a list of dicts
                if isinstance(content, str):
                    # Handle simple string content (same as before)
                    text_value = content
                    video_links = video_md_pattern.findall(text_value)
                    clean_text = video_md_pattern.sub("", text_value).strip()
                    if clean_text:
                        new_content_list.append({"type": "text", "text": clean_text})
                    for link in video_links:
                        new_content_list.append(
                            {
                                "type": "video_url",
                                "video_url": {"url": link},
                            }
                        )
                elif isinstance(content, list):
                    # Handle list content (e.g., text and images)
                    for element in content:
                        if element.get("type") == "text":
                            text_value = element.get("text", "")
                            video_links = video_md_pattern.findall(text_value)
                            clean_text = video_md_pattern.sub("", text_value).strip()
                            if clean_text:
                                new_content_list.append({"type": "text", "text": clean_text})
                            for link in video_links:
                                new_content_list.append(
                                    {
                                        "type": "video_url",
                                        "video_url": {"url": link},
                                    }
                                )
                        else:
                            # Pass through non-text elements (e.g., images) unchanged
                            new_content_list.append(element)
                else:
                     logger.warning(f"Unexpected content type in message: {type(content)}. Skipping.")
                     continue # Skip this message or handle appropriately


                # Reconstruct the message: if content became a list with only one text item, simplify back to string
                if len(new_content_list) == 1 and new_content_list[0].get("type") == "text":
                    processed_messages.append({"role": role, "content": new_content_list[0]["text"]})
                elif len(new_content_list) > 0 : # Use list for multiple parts or non-text parts
                    processed_messages.append({"role": role, "content": new_content_list})

            # Optional retrieval-augmented context (applied after processing)
            nearest_items = prompt_item.prompts[-1].metadata.get("nearestItems", [])
            if nearest_items:
                context = prompt_item.build_context(nearest_items=nearest_items, add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                # Append context as assistant message
                processed_messages.append({"role": "assistant", "content": context})

            # Call model and stream/collect the response
            stream_response = self.call_model(messages=processed_messages)
            accumulated_response = ""
            for chunk in stream_response:
                accumulated_response += chunk
                prompt_item.add(
                    message={
                        "role": "assistant",
                        "content": [
                            {
                                "mimetype": dl.PromptType.TEXT,
                                "value": accumulated_response,
                            }
                        ],
                    },
                    model_info={
                        "name": model_name,
                        "confidence": 1.0,
                        "model_id": self.model_entity.id,
                    },
                )
        # Return an empty list to conform to the Dataloop SDK expectations
        return []
