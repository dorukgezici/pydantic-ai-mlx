<div align="center">
	<h1 align="center">pydantic-ai-mlx</h1>
	<p align="center">MLX local inference for <a href="https://github.com/pydantic/pydantic-ai" target="_blank">Pydantic AI</a> through <a href="https://github.com/lmstudio-ai/mlx-engine" target="_blank">LM Studio</a> or <a href="https://github.com/ml-explore/mlx-examples/blob/main/llms" target="_blank">mlx-lm</a> directly.</p>
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

Two options are provided as backends;
- LM Studio backend (OpenAI compatible server that can also utilize mlx-lm, model runs on a seperate process)
- mlx-lm backend (direct integration with Apple's library, model runs within your Python process, *experimental*)

*STILL IN DEVELOPMENT, NOT RECOMMENDED FOR PRODUCTION USE YET.*

Contributions are welcome!

### Features
- [x] LM Studio backend, should be fully supported
- [x] Streaming text support for mlx-lm backend
- [ ] Tool calling support for mlx-lm backend

_Apple's MLX seems more performant on Apple silicon than llama.cpp (Ollama), as of January 2025._

## Installation

```bash
uv add pydantic-ai-mlx
```

## Usage

### LM Studio backend
```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_lm_studio import LMStudioModel

model = LMStudioModel(model_name="mlx-community/Qwen2.5-7B-Instruct-4bit") # supports tool calling
agent = Agent(model, system_prompt="You are a chatbot.")

async def stream_response(user_prompt: str, message_history: list[ModelMessage]):
    async with agent.run_stream(user_prompt, message_history) as result:
        async for message in result.stream():
            yield message
```

### mlx-lm backend
```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_mlx_lm import MLXModel

model = MLXModel(model_name="mlx-community/Llama-3.2-3B-Instruct-4bit")
# See https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models
# also https://huggingface.co/mlx-community

agent = Agent(model, system_prompt="You are a chatbot.")

async def stream_response(user_prompt: str, message_history: list[ModelMessage]):
    async with agent.run_stream(user_prompt, message_history) as result:
        async for message in result.stream():
            yield message
```
