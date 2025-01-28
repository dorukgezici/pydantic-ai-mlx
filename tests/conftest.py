from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent, Tool

from pydantic_ai_mlx_lm import MLXModel

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"


@pytest.fixture
def m() -> MLXModel:
    return MLXModel(MODEL_NAME)


@pytest.fixture
def agent(m: MLXModel) -> Agent:
    return Agent(m, system_prompt="You are a chatbot, respond with a short and concise answer.")


@pytest.fixture
def agent_joker(m: MLXModel, agent: Agent) -> Agent:
    async def generate_joke() -> str:
        """Generate a short joke.

        Returns:
            str: A short joke.
        """
        result = await agent.run("Tell me a short joke")
        return result.data

    return Agent(
        m,
        system_prompt="You are a chatbot who loves to tell jokes.",
        retries=5,
        tools=[Tool(generate_joke, takes_ctx=False)],
    )
