from mlx_lm.utils import GenerationResponse
from openai.types import chat
from pydantic_ai import _utils, usage
from pydantic_ai.messages import ToolCallPart


def map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
        id=_utils.guard_tool_call_id(t=t, model_source="mlx-lm"),
        type="function",
        function={"name": t.tool_name, "arguments": t.args_as_json_str()},
    )


def map_usage(response: GenerationResponse) -> usage.Usage:
    return usage.Usage(
        request_tokens=response.prompt_tokens,
        response_tokens=response.generation_tokens,
        total_tokens=response.prompt_tokens + response.generation_tokens,
        details=None,
    )
