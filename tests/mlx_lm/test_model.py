from __future__ import annotations as _annotations

from inline_snapshot import snapshot
from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import Usage

from pydantic_ai_mlx_lm import MLXModel


def test_init(model: MLXModel):
    assert isinstance(model.model, Module)
    assert isinstance(model.tokenizer, TokenizerWrapper)
    assert model.name() == f"mlx-lm:{model.model_name}"


async def test_agent(agent: Agent):
    result = await agent.run("How many states are there in USA?")
    messages = result.new_messages()

    assert result.usage() == snapshot(
        Usage(requests=1, request_tokens=25, response_tokens=385, total_tokens=410),
    )

    assert isinstance(messages[-1], ModelResponse)
    assert isinstance(messages[-1].parts[0], TextPart)
    assert messages[-1].parts[0].content == result.data

    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.data == snapshot(
        "There are 50 states in the United States of America.",
    )


async def test_agent_stream(agent: Agent):
    async with agent.run_stream("Who is the current president of USA?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=0.1)] == snapshot(
            ["As of my knowledge cutoff in 2023, Joe Biden is the President of the United States."],
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=1176, response_tokens=231, total_tokens=1407),
        )
