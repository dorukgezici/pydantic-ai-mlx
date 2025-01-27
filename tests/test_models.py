from __future__ import annotations as _annotations

from devtools import debug
from inline_snapshot import snapshot
from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import Usage

from pydantic_ai_mlx_lm import MLXModel


def test_init(m: MLXModel):
    assert isinstance(m.model, Module)
    assert isinstance(m.tokenizer, TokenizerWrapper)
    assert m.name() == f"mlx-lm:{m.model_name}"


async def test_request_simple_success(agent: Agent):
    result = await agent.run("How many states are there in USA?")
    messages = result.new_messages()

    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.usage() == snapshot(Usage(requests=1))

    assert isinstance(messages[-1], ModelResponse)
    assert isinstance(messages[-1].parts[0], TextPart)
    assert messages[-1].parts[0].content == snapshot("There are 50 states in the United States of America.")


async def test_request_tools_success(agent_with_tools: Agent):
    result = await agent_with_tools.run("My name is Doruk.")
    messages = result.new_messages()

    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.usage() == snapshot(Usage(requests=1))
    debug(messages)
    assert isinstance(messages[-1], ModelResponse)
    assert isinstance(messages[-1].parts[0], TextPart)
    assert messages[-1].parts[0].content == snapshot(
        "Hello Doruk! It's nice to meet you. Is there something I can help you with, or would you like to chat?"
    )


async def test_stream_text(agent: Agent):
    async with agent.run_stream("Who is the current president of USA?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=0.1)] == snapshot(
            [
                "As of my knowledge cutoff in December 2023, Joe Biden is the President of the United States. However, please note that my information may not be up-to-date, and you may want to verify this information for the most recent and accurate answer."
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=2236, response_tokens=1378, total_tokens=3614)
        )
