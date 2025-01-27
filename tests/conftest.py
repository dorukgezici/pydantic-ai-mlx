from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent, Tool

from pydantic_ai_mlx_lm import MLXModel

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
# MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture
def m() -> MLXModel:
    return MLXModel(MODEL_NAME)


@pytest.fixture
def agent(m: MLXModel) -> Agent:
    return Agent(m, system_prompt="You are a chatbot, respond with a short and concise answer.")


@pytest.fixture
def agent_with_tools(m: MLXModel) -> Agent:
    def greet_user(name: str) -> str:
        """Greet the user with their name.

        Args:
            name (str): Name of the user.

        Returns:
            str: Greeting message.
        """
        return f"Hey there {name}!"

    return Agent(
        model=m,
        system_prompt="You are a chatbot, just greet the user.",
        tools=[
            Tool(greet_user, takes_ctx=False),
        ],
    )
