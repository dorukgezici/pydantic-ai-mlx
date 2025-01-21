from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterable, AsyncIterator

from openai.types import chat
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import StreamedResponse


@dataclass
class MLXStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for MLX models."""

    _response: AsyncIterable[chat.ChatCompletionChunk]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # Handle the text part of the response
            content = choice.delta.content
            if content is not None:
                yield self._parts_manager.handle_text_delta(vendor_part_id="content", content=content)

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    def timestamp(self) -> datetime:
        return self._timestamp
