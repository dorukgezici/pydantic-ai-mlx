from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent, Tool

from pydantic_ai_lm_studio import LMStudioModel


@pytest.fixture
def model_name() -> str:
    return "mlx-community/Qwen2.5-7B-Instruct-1M-6bit"


@pytest.fixture
def model(model_name: str) -> LMStudioModel:
    return LMStudioModel(model_name)


@pytest.fixture
def agent(model: LMStudioModel) -> Agent:
    return Agent(model, system_prompt="You are a chatbot, respond short and concisely.")


@pytest.fixture
def agent_joker(model: LMStudioModel, agent: Agent) -> Agent:
    async def generate_joke() -> str:
        """Generate a short joke.

        Returns:
            str: Generated joke.
        """
        result = await agent.run("Tell me a short joke.")
        return result.data

    return Agent(
        model,
        system_prompt="You are a chatbot who loves to tell jokes.",
        tools=[Tool(generate_joke, takes_ctx=False)],
    )
