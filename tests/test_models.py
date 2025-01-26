from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot
from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import Usage

from pydantic_ai_mlx_lm import MLXModel

MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


@pytest.fixture
def m() -> MLXModel:
    return MLXModel(MODEL_NAME)


@pytest.fixture
def agent(m: MLXModel) -> Agent:
    return Agent(m, system_prompt="You are a chatbot, please respond to the user.")


def test_init(m: MLXModel):
    assert isinstance(m.model, Module)
    assert isinstance(m.tokenizer, TokenizerWrapper)
    assert m.name() == f"mlx-lm:{MODEL_NAME}"


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
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            [
                "As",
                "As of",
                "As of my",
                "As of my last",
                "As of my last update",
                "As of my last update,",
                "As of my last update, the",
                "As of my last update, the current",
                "As of my last update, the current President",
                "As of my last update, the current President of",
                "As of my last update, the current President of the",
                "As of my last update, the current President of the United",
                "As of my last update, the current President of the United States",
                "As of my last update, the current President of the United States is",
                "As of my last update, the current President of the United States is Joe",
                "As of my last update, the current President of the United States is Joe Biden",
                "As of my last update, the current President of the United States is Joe Biden.",
                "As of my last update, the current President of the United States is Joe Biden. He",
                "As of my last update, the current President of the United States is Joe Biden. He assumed",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January ",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 2",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20,",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, ",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 20",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 202",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021.",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However,",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information may",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information may have",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information may have changed",
                "As of my last update, the current President of the United States is Joe Biden. He assumed office on January 20, 2021. However, please verify from a reliable source as this information may have changed.",
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=1))
