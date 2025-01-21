from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot
from mlx.nn import Module  # pyright: ignore reportMissingTypeStubs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import Usage

from pydantic_ai_mlx_lm import MLXModel

MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


@pytest.fixture
def m() -> MLXModel:
    return MLXModel(MODEL_NAME)


def test_init(m: MLXModel):
    assert isinstance(m.model, Module)
    assert isinstance(m.tokenizer, TokenizerWrapper)
    assert m.name() == f"mlx-lm:{MODEL_NAME}"


async def test_request_simple_success(m: MLXModel):
    agent = Agent(m, system_prompt="You are a chatbot, please respond to the user.")
    result = await agent.run("How many states are there in the USA?")
    messages = result.new_messages()

    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.usage() == snapshot(Usage(requests=1))

    assert isinstance(messages[-1], ModelResponse)
    assert isinstance(messages[-1].parts[0], TextPart)
    assert messages[-1].parts[0].content == snapshot(result.data)
