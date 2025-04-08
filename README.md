# VILA Model Adapter for Dataloop

This Dataloop model adapter integrates the VILA (Vision-Language) model, enabling advanced multimodal capabilities for processing text, images, and videos in Dataloop platform.

## Overview

VILA (Vision-Language) is a family of open vision-language models designed for efficiency and accuracy in image and video understanding. It supports:

- Image description and understanding
- Video captioning
- Multi-image reasoning
- In-context learning

This adapter allows you to use VILA models within the Dataloop platform for AI-assisted annotation, content generation, and multimodal reasoning.

## Features

- Process both images and videos
- Generate detailed descriptions of visual content
- Answer questions about visual content
- Stream responses for real-time feedback
- Support for different conversation modes

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| model_path | Hugging Face path to the VILA model | Efficient-Large-Model/NVILA-15B |
| conv_mode | Conversation mode for the model | auto |
| max_tokens | Maximum number of tokens to generate | 512 |
| temperature | Controls randomness (0-1) | 0.2 |
| top_p | Controls diversity via nucleus sampling (0-1) | 0.9 |
| stream | Whether to stream responses | true |

## Available Models

- NVILA-15B / NVILA-15B-Lite
- NVILA-8B / NVILA-8B-Lite
- VILA1.5-3B, VILA1.5-8B, VILA1.5-13B, VILA1.5-40B

## Usage in Dataloop

1. Create a model in Dataloop
2. Select "VILA Chat Completion" as the model adapter
3. Configure the adapter with appropriate settings
4. Deploy the model
5. Use the model with prompt items containing text and images

## License

- The models are released under the Apache 2.0 license
- See the [VILA repository](https://github.com/NVlabs/VILA) for full license details