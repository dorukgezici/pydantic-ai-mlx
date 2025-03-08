from __future__ import annotations as _annotations

from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import Usage

from pydantic_ai_lm_studio import LMStudioModel


def test_init(model_name: str, model: LMStudioModel):
    assert model.model_name == model_name


async def test_stream_text(agent: Agent):
    async with agent.run_stream("Who is the current president of USA?") as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=0.1)] == snapshot(
            ["The current", "The current president of the USA", "The current president of the USA is Joe Biden."],
        )
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(requests=1, request_tokens=1197, response_tokens=231, total_tokens=1428),
        )


async def test_agent(agent: Agent):
    result = await agent.run("How many states are there in USA?")
    messages = result.new_messages()

    assert result.usage() == snapshot(
        Usage(requests=1, request_tokens=34, response_tokens=4, total_tokens=38),
    )

    assert isinstance(messages[-1], ModelResponse)
    assert isinstance(messages[-1].parts[0], TextPart)
    assert messages[-1].parts[0].content == result.data

    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.data == snapshot(
        "50 states.",
    )


async def test_agent_joker(agent_joker: Agent):
    result = await agent_joker.run("Hey! I am Doruk. Tell me a joke.")
    messages = result.new_messages()

    assert result.usage() == snapshot(
        Usage(requests=2, request_tokens=367, response_tokens=42, total_tokens=409),
    )

    # model decides to call tool
    assert isinstance(messages[1], ModelResponse)
    assert isinstance(messages[1].parts[0], ToolCallPart)
    assert messages[1].parts[0].tool_name == "generate_joke"

    # tool responds to model
    assert isinstance(messages[2], ModelRequest)
    assert isinstance(messages[2].parts[0], ToolReturnPart)
    assert messages[2].parts[0].tool_name == "generate_joke"
    assert messages[2].parts[0].content == snapshot("""\
Why don't skeletons fight each other?

They don't have the guts!\
""")

    # model's final response
    assert isinstance(messages[3], ModelResponse)
    assert isinstance(messages[3].parts[0], TextPart)
    assert messages[3].parts[0].content == result.data

    # result
    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert result.data == snapshot(
        "Sure, here's a joke for you: Why don't skeletons fight each other? They don't have the guts! 😄",
    )
