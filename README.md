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

Still in development:
- [x] Non-streaming and streaming text requests
- [ ] Tool support

_Apple's MLX seems more performant on Apple Silicon than llama.cpp, which is what Ollama uses. (as of January 25)_

## Installation

```bash
uv add pydantic-ai-mlx-lm
```
