from typing import Generator

from mlx_lm.utils import GenerationResponse


class AsyncGeneratorWrapper:
    def __init__(self, sync_gen: Generator[GenerationResponse, None, None]):
        self._sync_gen = sync_gen

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._sync_gen)
        except StopIteration:
            raise StopAsyncIteration
