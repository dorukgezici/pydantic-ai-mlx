from typing import Iterable

from openai.types import chat
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def map_message(message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
    """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""

    if isinstance(message, ModelRequest):
        yield from _map_user_message(message)

    if isinstance(message, ModelResponse):
        texts: list[str] = []
        tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
        for item in message.parts:
            if isinstance(item, TextPart):
                texts.append(item.content)

            if isinstance(item, ToolCallPart):
                tool_calls.append(_map_tool_call(item))

        message_param = chat.ChatCompletionAssistantMessageParam(role="assistant")
        if texts:
            # Note: model responses from this model should only have one text item, so the following
            # shouldn't merge multiple texts into one unless you switch models between runs:
            message_param["content"] = "\n\n".join(texts)
        if tool_calls:
            message_param["tool_calls"] = tool_calls
        yield message_param


def _map_user_message(message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            pass
            # yield chat.ChatCompletionSystemMessageParam(role="system", content=part.content)

        if isinstance(part, UserPromptPart):
            yield chat.ChatCompletionUserMessageParam(role="user", content=part.content)

        if isinstance(part, ToolReturnPart):
            yield chat.ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=_guard_tool_call_id(t=part, model_source="OpenAI"),
                content=part.model_response_str(),
            )

        if isinstance(part, RetryPromptPart):
            if part.tool_name is None:
                yield chat.ChatCompletionUserMessageParam(role="user", content=part.model_response())
            else:
                yield chat.ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=_guard_tool_call_id(t=part, model_source="OpenAI"),
                    content=part.model_response(),
                )


def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
        id=_guard_tool_call_id(t=t, model_source="OpenAI"),
        type="function",
        function={"name": t.tool_name, "arguments": t.args_as_json_str()},
    )
