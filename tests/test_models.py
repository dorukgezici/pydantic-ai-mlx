from __future__ import annotations as _annotations

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
    assert messages[-1].parts[0].content == snapshot(
        "As of 2021, there are 50 states in the United States of America. The 50 states are: Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, Florida, Georgia, Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri, Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming. The 50th and most recent state to join the Union was Hawaii in 1959."
    )


async def test_stream_text(agent: Agent):
    async with agent.run_stream("Who is the current president of USA?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=0.1)] == snapshot(
            [
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information may have changed."
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=624, response_tokens=1176, total_tokens=1800)
        )
