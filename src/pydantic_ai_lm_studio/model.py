from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from pydantic_ai.models.openai import OpenAIModel

from pydantic_ai_lm_studio.provider import LMStudioProvider

LMStudioModelName = Union[
    str,
    Literal[
        "mlx-community/Qwen2.5-7B-Instruct-1M-6bit",  # tool support
        "mlx-community/Llama-3.2-3B-Instruct-8bit",  # tool support
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    ],
]


@dataclass(init=False)
class LMStudioModel(OpenAIModel):
    """A model that implements LM Studio using the OpenAI API."""

    _model_name: LMStudioModelName = field(repr=False)

    def __init__(
        self,
        model_name: LMStudioModelName,
        *,
        provider: LMStudioProvider = LMStudioProvider(),
        **kwargs: dict[str, Any],
    ):
        super().__init__(model_name=model_name, provider=provider, *kwargs)
