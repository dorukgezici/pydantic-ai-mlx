# from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Generator, Literal, overload

from devtools import debug

# from typing import AsyncIterator, Generator, Iterable, Literal, overload
from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, generate, stream_generate  # pyright: ignore[reportUnknownVariableType]
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
)

# from pydantic_ai.models import AgentModel, StreamedResponse
from pydantic_ai.models import AgentModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

from .utils import map_message


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

    # @asynccontextmanager
    # async def request_stream(
    #     self,
    #     messages: list[ModelMessage],
    #     model_settings: ModelSettings | None,
    # ) -> AsyncIterator[StreamedResponse]:
    #     """Make a streaming request to the model."""

    #     response = self._completions_create(messages, True, model_settings)
    #     yield await self._process_streamed_response(response)

    @overload
    def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[False], model_settings: ModelSettings | None
    ) -> str:
        pass

    @overload
    def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[True], model_settings: ModelSettings | None
    ) -> Generator[GenerationResponse, None, None]:
        pass

    def _completions_create(
        self, messages: list[ModelMessage], stream: bool, model_settings: ModelSettings | None
    ) -> str | Generator[GenerationResponse, None, None]:
        """Standalone function to make it easier to override"""

        conversation = list(chain(*(map_message(m) for m in messages)))
        debug(conversation)

        if not stream:
            return generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=self.tokenizer.apply_chat_template(  # pyright: ignore reportUnknownMemberType
                    conversation=conversation,
                    add_generation_prompt=True,
                ),
                max_tokens=model_settings.get("max_tokens", 1000) if model_settings else 1000,
            )

        return stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.tokenizer.apply_chat_template(  # pyright: ignore reportUnknownMemberType
                conversation=conversation,
                add_generation_prompt=True,
            ),
            max_tokens=model_settings.get("max_tokens", 1000) if model_settings else 1000,
        )

    @staticmethod
    def _process_response(response: str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""

        return ModelResponse(parts=[TextPart(content=response)], timestamp=datetime.now(timezone.utc))

    # @staticmethod
    # async def _process_streamed_response(response: Generator[GenerationResponse, None, None]) -> MLXStreamedResponse:
    #     """Process a streamed response, and prepare a streaming response to return."""

    #     peekable_response = _utils.PeekableAsyncStream(response)
    #     first_chunk = await peekable_response.peek()
    #     if isinstance(first_chunk, _utils.Unset):
    #         raise UnexpectedModelBehavior("Streamed response ended without content or tool calls")

    #     return MLXStreamedResponse(peekable_response, datetime.fromtimestamp(first_chunk.created, tz=timezone.utc))
