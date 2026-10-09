"""Async HTTP clients for vLLM completions and the Trace-OPSD tool service."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

import httpx

from trace_opsd.tools.python_executor import (
    PYTHON_EXECUTION_RESULT_VERSION,
    classify_python_error,
)

from .config import ModelConfig, RuntimeConfig, ToolsConfig
from .models import (
    GenerationRequest,
    GenerationResult,
    ToolCall,
    ToolResult,
)
from .token_history import exact_token_ids


@runtime_checkable
class GenerationClient(Protocol):
    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]: ...


@runtime_checkable
class ToolClient(Protocol):
    async def execute_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]: ...


class _HTTPResponseError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        # Let ``generate_batch`` immediately requeue OOM responses behind the
        # lower fallback-concurrency gate instead of hammering the server at
        # the original concurrency for every transport retry.
        if _is_oom_error(self.body):
            return False
        return self.status in {408, 409, 425, 429} or self.status >= 500


class _TransportError(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class _RetryingJSONClient:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        concurrency: int,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    limits=httpx.Limits(
                        max_connections=self._concurrency,
                        max_keepalive_connections=self._concurrency,
                        keepalive_expiry=30.0,
                    ),
                )
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "_RetryingJSONClient":
        await self._get_client()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Any, int]:
        attempts = self.max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    client = await self._get_client()
                    response = await client.post(
                        url,
                        json=dict(payload),
                        headers=dict(headers or {}),
                    )
                if not 200 <= response.status_code < 300:
                    raise _HTTPResponseError(response.status_code, response.text)
                try:
                    return response.json(), attempt
                except ValueError as exc:
                    raise RuntimeError("HTTP response was not valid JSON") from exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                retryable = not isinstance(exc, _HTTPResponseError) or exc.retryable
                if attempt >= attempts or not retryable:
                    break
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                if delay:
                    await asyncio.sleep(delay)
        message = f"{type(last_error).__name__}: {last_error}"
        raise _TransportError(message, min(attempts, attempt)) from last_error


class VLLMOpenAICompletionsClient(_RetryingJSONClient):
    """Call vLLM's OpenAI-compatible ``/v1/completions`` endpoint."""

    def __init__(self, model: ModelConfig, runtime: RuntimeConfig) -> None:
        super().__init__(
            timeout_seconds=runtime.generation_timeout_seconds,
            max_retries=runtime.generation_max_retries,
            retry_backoff_seconds=runtime.generation_retry_backoff_seconds,
            concurrency=runtime.request_concurrency,
        )
        self.model = model
        self.fallback_concurrency = runtime.fallback_concurrency
        self._fallback_gate = asyncio.Semaphore(runtime.fallback_concurrency)
        self._oom_fallback_active = False
        self.endpoint = f"{model.base_url.rstrip('/')}/completions"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "stop": list(request.stop),
            "stream": False,
            "echo": False,
        }
        # vLLM-specific OpenAI extensions. In the official OpenAI SDK these
        # values would be supplied through ``extra_body``; raw HTTP merges
        # them into the JSON body directly.
        extra_body = {
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "ignore_eos": request.ignore_eos,
            "include_stop_str_in_output": request.include_stop_str_in_output,
            **dict(request.extra_body),
        }
        payload.update(extra_body)
        if request.prompt_token_ids is not None:
            payload["prompt"] = exact_token_ids(
                request.prompt_token_ids, name="completion prompt"
            )
            payload["return_token_ids"] = True
        headers = {"Authorization": f"Bearer {self.model.api_key}"}
        try:
            raw, attempts = await self._post_json(
                self.endpoint, payload, headers=headers
            )
            if not isinstance(raw, Mapping):
                raise ValueError("completion response must be a JSON object")
            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("completion response did not contain choices")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ValueError("completion choice must be a JSON object")
            token_ids = choice.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    raise ValueError("completion token_ids must be a list")
                token_ids = tuple(exact_token_ids(token_ids, name="completion output"))
            text = choice.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            usage = raw.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            return GenerationResult(
                request_id=request.request_id,
                text=text,
                finish_reason=(
                    str(choice["finish_reason"])
                    if choice.get("finish_reason") is not None
                    else None
                ),
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                completion_tokens=_optional_int(usage.get("completion_tokens")),
                raw_response=dict(raw),
                attempts=attempts,
                latency_ms=(time.perf_counter() - started) * 1000,
                token_ids=token_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return GenerationResult(
                request_id=request.request_id,
                raw_response=None,
                error=f"{type(exc).__name__}: {exc}",
                attempts=getattr(exc, "attempts", 1),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]:
        if not requests:
            return []

        async def generate_with_current_limit(
            request: GenerationRequest,
        ) -> GenerationResult:
            if not self._oom_fallback_active:
                return await self.generate(request)
            async with self._fallback_gate:
                return await self.generate(request)

        results = list(
            await asyncio.gather(
                *(generate_with_current_limit(item) for item in requests)
            )
        )
        oom_indices = [
            index
            for index, result in enumerate(results)
            if result.error is not None and _is_oom_error(result.error)
        ]
        if not oom_indices:
            return results

        self._oom_fallback_active = True

        async def retry(index: int) -> tuple[int, GenerationResult]:
            async with self._fallback_gate:
                retried = await self.generate(requests[index])
            original = results[index]
            return index, replace(
                retried,
                attempts=original.attempts + retried.attempts,
                latency_ms=(original.latency_ms or 0.0) + (retried.latency_ms or 0.0),
            )

        for index, retried in await asyncio.gather(
            *(retry(index) for index in oom_indices)
        ):
            results[index] = retried
        return results


class RoutedVLLMClient:
    """Keep every turn of a trajectory on one vLLM replica."""

    def __init__(self, model: ModelConfig, runtime: RuntimeConfig) -> None:
        if not model.replica_base_urls:
            raise ValueError("replica_base_urls must contain at least one endpoint")
        self.clients = tuple(
            VLLMOpenAICompletionsClient(
                model.model_copy(update={"base_url": base_url}), runtime
            )
            for base_url in model.replica_base_urls
        )

    def replica_index(self, request_id: str) -> int:
        trajectory_id = request_id.rsplit(":turn:", 1)[0]
        digest = hashlib.sha256(trajectory_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self.clients)

    async def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]:
        if not requests:
            return []
        grouped: list[list[tuple[int, GenerationRequest]]] = [[] for _ in self.clients]
        for index, request in enumerate(requests):
            grouped[self.replica_index(request.request_id)].append((index, request))
        batches = await asyncio.gather(
            *(
                client.generate_batch([request for _, request in group])
                for client, group in zip(self.clients, grouped)
            )
        )
        ordered: list[GenerationResult | None] = [None] * len(requests)
        for group, results in zip(grouped, batches):
            if len(group) != len(results):
                raise RuntimeError("vLLM replica returned the wrong number of results")
            for (index, _), result in zip(group, results):
                ordered[index] = result
        return [result for result in ordered if result is not None]

    async def aclose(self) -> None:
        await asyncio.gather(*(client.aclose() for client in self.clients))


class ToolHTTPBatchClient(_RetryingJSONClient):
    """Fan a batch of tool calls out to the unified Trace-OPSD service."""

    def __init__(self, tools: ToolsConfig, runtime: RuntimeConfig) -> None:
        super().__init__(
            timeout_seconds=tools.request_timeout_seconds,
            max_retries=tools.max_retries,
            retry_backoff_seconds=tools.retry_backoff_seconds,
            concurrency=runtime.request_concurrency,
        )
        self.tools = tools

    async def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        endpoint = self.tools.endpoints.get(call.tool_name)
        if endpoint is None or call.tool_name not in self.tools.enabled:
            error = f"unknown or disabled tool {call.tool_name!r}"
            return ToolResult(
                request_id=call.request_id,
                tool_name=call.tool_name,
                observation=_error_observation(error),
                status="error",
                error=error,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_origin="sandbox",
                execution_result_version=PYTHON_EXECUTION_RESULT_VERSION
                if call.tool_name == "python"
                else None,
            )
        url = f"{self.tools.base_url.rstrip('/')}{endpoint}"
        try:
            raw, attempts = await self._post_json(url, {"raw_text": call.raw_text})
            if not isinstance(raw, Mapping) or "result" not in raw:
                raise ValueError("tool response did not contain result")
            observation = raw["result"]
            if not isinstance(observation, str):
                observation = str(observation)
            structured = (
                call.tool_name == "python"
                and raw.get("execution_result_version")
                == PYTHON_EXECUTION_RESULT_VERSION
            )
            if structured and raw.get("status") not in {"ok", "error"}:
                raise ValueError("structured Python result has invalid status")
            logical_error = (
                raw["status"] == "error"
                if structured
                else (
                    raw.get("status") == "error"
                    or _observation_is_error(observation)
                    or (
                        call.tool_name == "python"
                        and classify_python_error(observation) is not None
                    )
                )
            )
            return ToolResult(
                request_id=call.request_id,
                tool_name=call.tool_name,
                observation=observation,
                status="error" if logical_error else "ok",
                raw_response=dict(raw),
                error=observation if logical_error else None,
                attempts=attempts,
                latency_ms=(time.perf_counter() - started) * 1000,
                error_category=(
                    raw.get("error_category")
                    if structured
                    else (
                        classify_python_error(observation)
                        if call.tool_name == "python" and logical_error
                        else None
                    )
                ),
                execution_result_version=raw.get("execution_result_version"),
                error_origin=raw.get("error_origin"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return ToolResult(
                request_id=call.request_id,
                tool_name=call.tool_name,
                observation=_error_observation(error),
                status="error",
                error=error,
                attempts=getattr(exc, "attempts", 1),
                latency_ms=(time.perf_counter() - started) * 1000,
                transport_error=True,
                error_origin="transport",
                execution_result_version=PYTHON_EXECUTION_RESULT_VERSION
                if call.tool_name == "python"
                else None,
            )

    async def execute_batch(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        if not calls:
            return []
        started = time.perf_counter()
        url = f"{self.tools.base_url.rstrip('/')}{self.tools.batch_endpoint}"
        payload = {
            "calls": [
                {
                    "request_id": call.request_id,
                    "tool": call.tool_name,
                    "raw_text": call.raw_text,
                }
                for call in calls
            ]
        }
        try:
            raw, attempts = await self._post_json(url, payload)
            if not isinstance(raw, Mapping) or not isinstance(raw.get("results"), list):
                raise ValueError("batch tool response did not contain results")
            by_id: dict[str, Mapping[str, Any]] = {}
            positional: list[Mapping[str, Any]] = []
            for item in raw["results"]:
                if not isinstance(item, Mapping):
                    continue
                positional.append(item)
                if item.get("request_id") is not None:
                    by_id[str(item["request_id"])] = item
            response_has_ids = bool(by_id)

            output: list[ToolResult] = []
            for index, call in enumerate(calls):
                item = by_id.get(call.request_id)
                if item is None and not response_has_ids and index < len(positional):
                    item = positional[index]
                if item is None:
                    error = "batch tool response omitted this request"
                    output.append(
                        ToolResult(
                            request_id=call.request_id,
                            tool_name=call.tool_name,
                            observation=_error_observation(error),
                            status="error",
                            raw_response=raw,
                            error=error,
                            attempts=attempts,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            transport_error=True,
                            error_origin="transport",
                            execution_result_version=PYTHON_EXECUTION_RESULT_VERSION
                            if call.tool_name == "python"
                            else None,
                        )
                    )
                    continue
                observation = item.get("result", item.get("observation", ""))
                if not isinstance(observation, str):
                    observation = str(observation)
                explicit_error = item.get("error")
                structured = (
                    call.tool_name == "python"
                    and item.get("execution_result_version")
                    == PYTHON_EXECUTION_RESULT_VERSION
                )
                if structured and item.get("status") not in {"ok", "error"}:
                    raise ValueError("structured Python result has invalid status")
                logical_error = (
                    item["status"] == "error"
                    if structured
                    else (
                        explicit_error is not None
                        or str(item.get("status", "ok")).casefold() == "error"
                        or _observation_is_error(observation)
                    )
                )
                if not observation:
                    if structured and not logical_error:
                        observation = "(no output)"
                    else:
                        observation = _error_observation(
                            str(explicit_error or "empty tool result")
                        )
                        logical_error = True
                output.append(
                    ToolResult(
                        request_id=call.request_id,
                        tool_name=str(
                            item.get("tool", item.get("tool_name", call.tool_name))
                        ),
                        observation=observation,
                        status="error" if logical_error else "ok",
                        raw_response=dict(item),
                        error=(
                            str(explicit_error)
                            if logical_error and explicit_error is not None
                            else observation
                            if logical_error
                            else None
                        ),
                        attempts=attempts,
                        latency_ms=_optional_float(item.get("latency_ms"))
                        if item.get("latency_ms") is not None
                        else (time.perf_counter() - started) * 1000,
                        error_category=(
                            item.get("error_category")
                            if structured
                            else (
                                str(item["error_category"])
                                if item.get("error_category") is not None
                                else classify_python_error(observation)
                                if call.tool_name == "python" and logical_error
                                else None
                            )
                        ),
                        execution_result_version=item.get("execution_result_version"),
                        error_origin=item.get("error_origin"),
                    )
                )
            return output
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts = getattr(exc, "attempts", 1)
            return [
                ToolResult(
                    request_id=call.request_id,
                    tool_name=call.tool_name,
                    observation=_error_observation(error),
                    status="error",
                    error=error,
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    transport_error=True,
                    error_origin="transport",
                    execution_result_version=PYTHON_EXECUTION_RESULT_VERSION
                    if call.tool_name == "python"
                    else None,
                )
                for call in calls
            ]


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _observation_is_error(observation: str) -> bool:
    lowered = observation.casefold()
    return "<result>\nerror:" in lowered or lowered.lstrip().startswith("error:")


def _error_observation(error: str) -> str:
    safe = error.replace("</result>", "&lt;/result&gt;")
    return f"<result>\nError: {safe}\n</result>"


def _is_oom_error(error: str) -> bool:
    lowered = error.casefold()
    return "out of memory" in lowered or "cuda oom" in lowered or "oom" in lowered


# Short aliases retained for CLI code and fakes.
VLLMClient = VLLMOpenAICompletionsClient
ToolHTTPClient = ToolHTTPBatchClient


__all__ = [
    "GenerationClient",
    "ToolClient",
    "ToolHTTPBatchClient",
    "ToolHTTPClient",
    "VLLMClient",
    "VLLMOpenAICompletionsClient",
]
