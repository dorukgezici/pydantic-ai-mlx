from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypedDict, Union

from mlx.nn import Module  # pyright: ignore reportMissingTypeStubs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import generate, load  # pyright: ignore reportUnknownVariableType
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    AgentModel,
    Model,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

KnownMLXModelNames = Literal["mlx-community/Mistral-7B-Instruct-v0.3-4bit"]
"""
For a full list see [link](github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models).
"""

MLXModelName = Union[KnownMLXModelNames, str]
"""
Since `mlx-lm` supports lots of models, we explicitly list the most common models but allow any name in the type hints.
"""


class Message(TypedDict):
    """OpenAI-compatible standard message format."""

    content: str
    role: str


@dataclass(init=False)
class MLXModel(Model):
    """A model backend that implements `mlx-lm` for local inference.

    Wrapper around [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) to run MLX compatible models locally on Apple Silicon.
    """

    model_name: MLXModelName
    model: Module
    tokenizer: TokenizerWrapper

    def __init__(self, model_name: MLXModelName):
        """Initialize an MLX model.

        Args:
            model_name: The name of the MLX compatible model to use. List of models available at
                github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models
        """
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create an agent model for function calling."""

        check_allow_model_requests()
        return MLXAgentModel(
            model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            allow_text_result=allow_text_result,
            function_tools=function_tools,
            result_tools=result_tools,
        )

    def name(self) -> str:
        return f"mlx-lm:{self.model_name}"


@dataclass
class MLXAgentModel(AgentModel):
    """Implementation of `AgentModel` for MLX models."""

    model_name: str
    model: Module
    tokenizer: TokenizerWrapper

    allow_text_result: bool
    function_tools: list[ToolDefinition]
    result_tools: list[ToolDefinition]

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model."""

        # TODO: Implement streamed requests
        raise NotImplementedError("Streamed requests not supported by this MLXAgentModel")
        yield

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, Usage]:
        """Make a non-streaming request to the model."""

        response = await self._completions_create(messages, False, model_settings)
        return (
            ModelResponse(parts=[TextPart(content=response)], timestamp=datetime.now(timezone.utc)),
            Usage(),
        )

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
    ) -> str:
        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.tokenizer.apply_chat_template(  # pyright: ignore reportUnknownMemberType
                conversation=self._map_messages(messages),
                add_generation_prompt=True,
            ),
            max_tokens=model_settings.get("max_tokens", 1000) if model_settings else 1000,
        )

    @classmethod
    def _map_messages(cls, messages: list[ModelMessage]) -> list[Message]:
        """Transforms a `pydantic_ai.ModelMessage` list to a `Message` TypedDict list."""

        conversation: list[Message] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        conversation.append(Message(content=part.content, role="user"))
            else:
                for part in message.parts:
                    if isinstance(part, TextPart):
                        conversation.append(Message(content=part.content, role="assistant"))
                    else:
                        # TODO: Implement tool calls
                        pass

        return conversation
