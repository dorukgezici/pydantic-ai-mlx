<div align="center">
	<h1 align="center">pydantic-ai-mlx-lm</h1>
	<p align="center">MLX local inference for <a href="https://github.com/pydantic/pydantic-ai" target="_blank">Pydantic AI</a> through <a href="https://github.com/ml-explore/mlx-examples/blob/main/llms" target="_blank">mlx-lm</a></p>
  <br/>
</div>

<p align="center">
  <a href="https://pypi.org/project/pydantic-ai-mlx-lm">
    <img src="https://img.shields.io/pypi/pyversions/pydantic-ai-mlx-lm" alt="pydantic-ai-mlx-lm" />
  </a>
  <a href="https://pypi.org/project/pydantic-ai-mlx-lm">
    <img src="https://img.shields.io/pypi/dm/pydantic-ai-mlx-lm" alt="PyPI download count">
  </a>
</p>

Run MLX compatible HuggingFace models on Apple silicon locally with Pydantic AI.

*STILL IN DEVELOPMENT, NOT RECOMMENDED FOR PRODUCTION USE YET.*

Contributions are welcome!

### Features
- [x] Non-streaming and streaming text support
- [ ] Tool usage support

_Apple's MLX seems more performant on Apple silicon than llama.cpp (Ollama), as of January 2025._

## Installation

```bash
uv add pydantic-ai-mlx-lm
```

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_mlx_lm import MLXModel

model = MLXModel(model_name="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
# See https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models
# also https://huggingface.co/mlx-community

agent = Agent(model, system_prompt="You are a chatbot.")

async def stream_response(user_prompt: str, message_history: list[ModelMessage]):
    async with agent.run_stream(user_prompt, message_history) as result:
        async for message in result.stream():
            yield message
```
