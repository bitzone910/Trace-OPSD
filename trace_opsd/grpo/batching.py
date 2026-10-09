"""Process-local asynchronous micro-batching for Trace-OPSD tool calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Sequence

from trace_opsd.rollout.models import ToolCall, ToolResult


class BatchToolClient(Protocol):
    async def execute_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]: ...


@dataclass
class _PendingCall:
    call: ToolCall
    future: asyncio.Future[ToolResult]


def _error_result(call: ToolCall, error: str) -> ToolResult:
    safe = error.replace("</result>", "&lt;/result&gt;")
    return ToolResult(
        request_id=call.request_id,
        tool_name=call.tool_name,
        observation=f"<result>\nError: {safe}\n</result>",
        status="error",
        error=error,
        transport_error=True,
    )


class AsyncToolMicroBatcher:
    """Coalesce calls from concurrent trajectories into `/tools/batch` requests."""

    def __init__(
        self,
        client: BatchToolClient,
        *,
        max_batch_size: int = 128,
        max_wait_seconds: float = 0.010,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be non-negative")
        self.client = client
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._pending: list[_PendingCall] = []
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[None] | None = None
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def submit(self, call: ToolCall) -> ToolResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolResult] = loop.create_future()
        dispatch: list[_PendingCall] | None = None
        async with self._lock:
            if self._closed:
                raise RuntimeError("tool micro-batcher is closed")
            self._pending.append(_PendingCall(call=call, future=future))
            if len(self._pending) >= self.max_batch_size:
                dispatch = self._take_pending_locked()
            elif self._timer is None:
                self._timer = asyncio.create_task(self._flush_after_delay())
        if dispatch:
            self._start_dispatch(dispatch)
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _take_pending_locked(self) -> list[_PendingCall]:
        batch = self._pending[: self.max_batch_size]
        del self._pending[: self.max_batch_size]
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._pending:
            self._timer = asyncio.create_task(self._flush_after_delay())
        return batch

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self.max_wait_seconds)
            async with self._lock:
                self._timer = None
                batch = self._take_pending_locked() if self._pending else []
            if batch:
                self._start_dispatch(batch)
        except asyncio.CancelledError:
            return

    def _start_dispatch(self, batch: list[_PendingCall]) -> None:
        task = asyncio.create_task(self._dispatch(batch))
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch(self, batch: list[_PendingCall]) -> None:
        active = [item for item in batch if not item.future.cancelled()]
        if not active:
            return
        calls = [item.call for item in active]
        try:
            results = await self.client.execute_batch(calls)
            by_id = {result.request_id: result for result in results}
            for item in active:
                result = by_id.get(item.call.request_id)
                if result is None:
                    result = _error_result(
                        item.call, "batch tool response omitted this request"
                    )
                if not item.future.done():
                    item.future.set_result(result)
        except asyncio.CancelledError:
            for item in active:
                if not item.future.done():
                    item.future.cancel()
            raise
        except Exception as exc:
            error = f"tool batch failed ({type(exc).__name__}: {exc})"
            for item in active:
                if not item.future.done():
                    item.future.set_result(_error_result(item.call, error))

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            pending, self._pending = self._pending, []
        for item in pending:
            if not item.future.done():
                item.future.set_result(_error_result(item.call, "tool batcher closed"))
        if self._dispatch_tasks:
            await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)
        close = getattr(self.client, "aclose", None)
        if close is not None:
            value = close()
            if hasattr(value, "__await__"):
                await value


_SHARED_BATCHERS: dict[tuple[object, ...], AsyncToolMicroBatcher] = {}


def get_shared_micro_batcher(
    *,
    base_url: str,
    max_batch_size: int = 128,
    max_wait_seconds: float = 0.010,
    request_concurrency: int = 64,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
) -> AsyncToolMicroBatcher:
    """Return one micro-batcher per event loop/configuration in a Ray worker."""

    from trace_opsd.rollout.clients import ToolHTTPBatchClient
    from trace_opsd.rollout.config import RuntimeConfig, ToolsConfig

    loop = asyncio.get_running_loop()
    key = (
        id(loop),
        base_url.rstrip("/"),
        max_batch_size,
        int(max_wait_seconds * 1e6),
        request_concurrency,
        timeout_seconds,
        max_retries,
        retry_backoff_seconds,
    )
    existing = _SHARED_BATCHERS.get(key)
    if existing is not None:
        return existing
    tools = ToolsConfig(
        base_url=base_url,
        request_timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    runtime = RuntimeConfig(
        request_concurrency=request_concurrency,
        fallback_concurrency=min(32, request_concurrency),
    )
    batcher = AsyncToolMicroBatcher(
        ToolHTTPBatchClient(tools, runtime),
        max_batch_size=max_batch_size,
        max_wait_seconds=max_wait_seconds,
    )
    _SHARED_BATCHERS[key] = batcher
    return batcher


__all__ = ["AsyncToolMicroBatcher", "BatchToolClient", "get_shared_micro_batcher"]
