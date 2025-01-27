from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import AsyncIterable, AsyncIterator, Iterable, Literal, overload

from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, generate, stream_generate  # pyright: ignore[reportUnknownVariableType]
from openai.types import chat
from pydantic_ai import _utils
from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id
from pydantic_ai.exceptions import UnexpectedModelBehavior
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
from pydantic_ai.models import AgentModel, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

from .response import MLXStreamedResponse
from .stream import AsyncStream
from .utils import map_tool_call


@dataclass
class MLXAgentModel(AgentModel):
    """Implementation of `AgentModel` for MLX models."""

    model_name: str
    model: Module
    tokenizer: TokenizerWrapper

    allow_text_result: bool
    function_tools: list[ToolDefinition]
    result_tools: list[ToolDefinition]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, Usage]:
        """Make a non-streaming request to the model."""

        response = self._completions_create(messages, False, model_settings)
        return self._process_response(response), Usage()

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model."""

        response = self._completions_create(messages, True, model_settings)
        yield await self._process_streamed_response(response)

    @overload
    def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[False], model_settings: ModelSettings | None
    ) -> str:
        pass

    @overload
    def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[True], model_settings: ModelSettings | None
    ) -> AsyncIterable[GenerationResponse]:
        pass

    def _completions_create(
        self, messages: list[ModelMessage], stream: bool, model_settings: ModelSettings | None
    ) -> str | AsyncIterable[GenerationResponse]:
        """Standalone function to make it easier to override"""

        conversation = list(chain(*(self._map_message(m) for m in messages)))
        prompt = self.tokenizer.apply_chat_template(  # type: ignore
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not stream:
            return generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,  # type: ignore
                max_tokens=model_settings.get("max_tokens", 1000) if model_settings else 1000,
            )

        generator = stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,  # type: ignore
            max_tokens=model_settings.get("max_tokens", 1000) if model_settings else 1000,
        )
        return AsyncStream(generator)

    def _process_response(self, response: str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""

        return ModelResponse(
            parts=[TextPart(content=response)],
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc),
        )

    async def _process_streamed_response(self, response: AsyncIterable[GenerationResponse]) -> MLXStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""

        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()

        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior("Streamed response ended without content or tool calls")

        return MLXStreamedResponse(
            _model_name=self.model_name,
            _response=peekable_response,
            _timestamp=datetime.now(timezone.utc),
        )

    def _map_message(self, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""

        if isinstance(message, ModelRequest):
            yield from self._map_user_message(message)

        if isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)

                if isinstance(item, ToolCallPart):
                    tool_calls.append(map_tool_call(item))

            message_param = chat.ChatCompletionAssistantMessageParam(role="assistant")
            if texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param["content"] = "\n\n".join(texts)
            if tool_calls:
                message_param["tool_calls"] = tool_calls
            yield message_param

    def _map_user_message(self, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
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
