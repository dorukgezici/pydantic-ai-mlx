from dataclasses import dataclass
from typing import Any, Generic, Iterator, Self, TypeVar

from pydantic_ai import _utils

T = TypeVar("T")


@dataclass
class AsyncStream(Generic[T]):
    """Wraps a synchronous iterator in an asynchronous interface.

    This class allows a synchronous iterator to be treated as an
    asynchronous iterator, enabling iteration in `async for` loops
    and usage within `async with` blocks.

    Example usage:
        async def example():
            sync_iter = iter([1, 2, 3])
            async_stream = AsyncStream(sync_iter)

            async for item in async_stream:
                print(item)

            async with AsyncStream(sync_iter) as stream:
                async for item in stream:
                    print(item)
    """

    _iter: Iterator[T]
    """The underlying synchronous iterator."""

    async def __anext__(self) -> T:
        """Return the next item from the synchronous iterator as if it were asynchronous.

        Calls `_utils.sync_anext` to retrieve the next item from the underlying
        synchronous iterator. If the iterator is exhausted, `StopAsyncIteration`
        is raised.
        """
        return _utils.sync_anext(self._iter)

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        pass
