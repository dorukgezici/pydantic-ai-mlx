from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal, Union

from mlx.nn import Module  # pyright: ignore[reportMissingTypeStubs]
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load  # pyright: ignore[reportUnknownVariableType]
from pydantic_ai.models import AgentModel, Model, check_allow_model_requests
from pydantic_ai.tools import ToolDefinition

from .agent_model import MLXAgentModel

KnownMLXModelName = Literal["mlx-community/Mistral-7B-Instruct-v0.3-4bit"]
"""
For a full list see [link](github.com/ml-explore/mlx-examples/blob/main/llms/README.md#supported-models).
"""

MLXModelName = Union[KnownMLXModelName, str]
"""
Since `mlx-lm` supports lots of models, we explicitly list the most common models but allow any name in the type hints.
"""


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
        self, *, function_tools: list[ToolDefinition], allow_text_result: bool, result_tools: list[ToolDefinition]
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
