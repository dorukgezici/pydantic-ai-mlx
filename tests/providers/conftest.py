from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel

from pydantic_ai_providers import LMStudioProvider, OpenRouterProvider

Provider = LMStudioProvider | OpenRouterProvider


@pytest.fixture
def provider() -> Provider:
    return LMStudioProvider()


@pytest.fixture
def model_name(provider: Provider) -> str:
    if isinstance(provider, LMStudioProvider):
        return "mlx-community/Qwen2.5-7B-Instruct-1M-6bit"
    else:
        return "google/gemini-2.0-flash-001"


@pytest.fixture
def model(model_name: str, provider: Provider) -> OpenAIModel:
    return OpenAIModel(model_name, provider=provider)


@pytest.fixture
def agent(model: OpenAIModel) -> Agent:
    return Agent(model, system_prompt="You are a chatbot, respond short and concisely.")


@pytest.fixture
def agent_joker(model: OpenAIModel, agent: Agent) -> Agent:
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
