from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from pydantic_ai_mlx_lm import MLXModel

MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
# MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture
def m() -> MLXModel:
    return MLXModel(MODEL_NAME)


@pytest.fixture
def agent(m: MLXModel) -> Agent:
    return Agent(m, system_prompt="You are a chatbot, respond with a short and concise answer.")
