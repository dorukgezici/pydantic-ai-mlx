from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from pydantic_ai_mlx_lm import MLXModel


@pytest.fixture
def model() -> MLXModel:
    return MLXModel("mlx-community/Llama-3.2-3B-Instruct-4bit")


@pytest.fixture
def agent(model: MLXModel) -> Agent:
    return Agent(model, system_prompt="You are a chatbot, respond short and concisely.")
