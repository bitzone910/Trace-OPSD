"""Tool registry, lifecycle, and total public call boundaries."""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from threading import RLock
from time import perf_counter
from typing import Callable

from .common import extract_tagged_content, wrap_result
from .config import ToolConfig
from .python_executor import (
    PYTHON_EXECUTION_RESULT_VERSION,
    PythonExecutionResult,
    PythonExecutor,
    find_dangerous_operation,
)
from .search_backend import (
    RetrieverSearchBackend,
    SearchBackend,
    format_search_results,
)


class ToolRuntime:
    """An initialized, concurrency-safe collection of rollout tools."""

    def __init__(
        self,
        config: ToolConfig,
        *,
        search_backend: SearchBackend,
        python_executor: PythonExecutor | None = None,
    ) -> None:
        self.config = config
        self.search_backend = search_backend
        self.python_executor = python_executor or PythonExecutor(config)
        self._registry: dict[str, Callable[[object], str]] = {
            "python": self.python_executor_tool,
            "search": self.search_tool,
        }

    @classmethod
    def from_config(cls, config: ToolConfig) -> "ToolRuntime":
        """Load one encoder and the General Search retriever exactly once."""

        from trace_opsd.retrieval.encoder import E5Encoder
        from trace_opsd.retrieval.general.retriever import DenseRetriever

        encoder = E5Encoder(
            config.model_path,
            device=config.device,
            max_length=config.max_length,
            use_fp16=config.use_fp16,
        )
        general = DenseRetriever(
            config.general_index_path,
            config.general_corpus_path,
            encoder=encoder,
            faiss_gpu=config.faiss_gpu,
            batch_size=config.retrieval_batch_size,
            corpus_num_proc=config.corpus_num_proc,
        )
        return cls(
            config,
            search_backend=RetrieverSearchBackend(general, database="general"),
        )

    def call_tool(self, tool_name: object, raw_text: object) -> str:
        try:
            if not isinstance(tool_name, str):
                return self._wrap("Error: tool_name must be a string.")
            tool = self._registry.get(tool_name)
            if tool is None:
                return self._wrap(f"Error: unknown tool '{tool_name}'.")
            return tool(raw_text)
        except BaseException as exc:
            return self._wrap(
                f"Error: tool dispatch failed ({type(exc).__name__}: {exc})"
            )

    def batch_call(self, calls: object) -> list[dict[str, object]]:
        """Execute a mixed batch without allowing one call to fail the batch.

        Search calls are grouped into vectorized backend requests. Python calls
        run concurrently up to the executor's configured global limit; the
        executor's own semaphore remains the final cross-request bound.
        """

        batch_started = perf_counter()
        if not isinstance(calls, list):
            return [
                self._batch_error(
                    None,
                    "Error: calls must be a list.",
                    batch_started,
                )
            ]
        if not calls:
            return []

        results: list[dict[str, object] | None] = [None] * len(calls)
        started_at = [perf_counter() for _ in calls]
        grouped: dict[str, list[tuple[int, object, object]]] = {
            name: [] for name in self._registry
        }

        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                results[index] = self._batch_error(
                    None,
                    "Error: each call must be an object.",
                    started_at[index],
                )
                continue

            request_id = call.get("request_id")
            if "request_id" not in call:
                results[index] = self._batch_error(
                    None,
                    "Error: call is missing request_id.",
                    started_at[index],
                )
                continue
            if "tool" not in call:
                results[index] = self._batch_error(
                    request_id,
                    "Error: call is missing tool.",
                    started_at[index],
                )
                continue
            if "raw_text" not in call:
                results[index] = self._batch_error(
                    request_id,
                    "Error: call is missing raw_text.",
                    started_at[index],
                )
                continue

            tool_name = call["tool"]
            if not isinstance(tool_name, str):
                results[index] = self._batch_error(
                    request_id,
                    "Error: tool must be a string.",
                    started_at[index],
                )
                continue
            if tool_name not in grouped:
                results[index] = self._batch_error(
                    request_id,
                    f"Error: unknown tool '{tool_name}'.",
                    started_at[index],
                )
                continue
            grouped[tool_name].append((index, request_id, call["raw_text"]))

        python_futures: dict[
            Future[PythonExecutionResult], tuple[int, object, float]
        ] = {}
        python_calls = grouped["python"]
        executor: ThreadPoolExecutor | None = None
        if python_calls:
            executor = ThreadPoolExecutor(
                max_workers=min(
                    len(python_calls),
                    self.config.python_max_concurrency,
                ),
                thread_name_prefix="trace_opsd-python-batch",
            )
            for index, request_id, raw_text in python_calls:
                future = executor.submit(self._python_executor_outcome, raw_text)
                python_futures[future] = (
                    index,
                    request_id,
                    started_at[index],
                )

        self._batch_search_calls(
            grouped["search"],
            tag="search",
            backend=self.search_backend,
            results=results,
            started_at=started_at,
        )

        try:
            for future in as_completed(python_futures):
                index, request_id, call_started = python_futures[future]
                try:
                    outcome = future.result()
                    results[index] = self._batch_result(
                        request_id,
                        outcome.observation,
                        outcome.status,
                        call_started,
                        error_category=outcome.error_category,
                        error_origin=outcome.error_origin,
                        execution_result_version=outcome.execution_result_version,
                    )
                except BaseException as exc:
                    outcome = self._python_error_outcome(
                        "sandbox_error",
                        f"Python batch dispatch failed ({type(exc).__name__}: {exc})",
                    )
                    results[index] = self._batch_result(
                        request_id,
                        outcome.observation,
                        outcome.status,
                        call_started,
                        error_category=outcome.error_category,
                        error_origin=outcome.error_origin,
                        execution_result_version=outcome.execution_result_version,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        for index, result in enumerate(results):
            if result is None:
                request_id = None
                call = calls[index]
                if isinstance(call, Mapping):
                    request_id = call.get("request_id")
                results[index] = self._batch_error(
                    request_id,
                    "Error: batch call did not produce a result.",
                    started_at[index],
                )
        return [result for result in results if result is not None]

    def python_executor_tool(self, raw_text: object) -> str:
        return self.python_executor_result(raw_text).observation

    def python_executor_result(self, raw_text: object) -> PythonExecutionResult:
        return self._python_executor_outcome(raw_text)

    def _python_executor_outcome(self, raw_text: object) -> PythonExecutionResult:
        try:
            input_error = self._validate_raw_text(raw_text)
            if input_error:
                return self._python_error_outcome("invalid_input", input_error)
            code = extract_tagged_content(raw_text, "python")
            if code is None:
                return self._python_error_outcome(
                    "invalid_input", "no Python code block detected"
                )
            if not code:
                return self._python_error_outcome("invalid_input", "empty code block")
            if len(code) > self.config.python_code_max_chars:
                return self._python_error_outcome(
                    "invalid_input", "Python code exceeds the configured length limit"
                )
            danger = find_dangerous_operation(code)
            if danger:
                return self._python_error_outcome(
                    "unsafe_code", f"restricted operation '{danger}' is not allowed"
                )
            outcome = self.python_executor.execute_result(code)
            return replace(outcome, observation=self._wrap(outcome.observation))
        except BaseException as exc:
            return self._python_error_outcome(
                "sandbox_error",
                f"Python Executor failed ({type(exc).__name__}: {exc})",
            )

    def _python_error_outcome(
        self, category: str, detail: object
    ) -> PythonExecutionResult:
        outcome = PythonExecutionResult.failure(category, detail, origin="sandbox")
        return replace(outcome, observation=self._wrap(outcome.observation))

    def search_tool(self, raw_text: object) -> str:
        return self._search_tool(
            raw_text,
            tag="search",
            backend=self.search_backend,
        )

    def _search_tool(
        self,
        raw_text: object,
        *,
        tag: str,
        backend: SearchBackend,
    ) -> str:
        try:
            query, error = self._prepare_search_query(raw_text, tag)
            if error is not None:
                return self._wrap(error)
            assert query is not None
            results = backend.search(query, top_k=self.config.search_top_k)
            return self._wrap(format_search_results(results))
        except BaseException as exc:
            return self._wrap(
                f"Error: search backend failed ({type(exc).__name__}: {exc})"
            )

    def _batch_search_calls(
        self,
        calls: list[tuple[int, object, object]],
        *,
        tag: str,
        backend: SearchBackend,
        results: list[dict[str, object] | None],
        started_at: list[float],
    ) -> None:
        prepared: list[tuple[int, object, str]] = []
        for index, request_id, raw_text in calls:
            query, error = self._prepare_search_query(raw_text, tag)
            if error is not None:
                results[index] = self._batch_error(
                    request_id,
                    error,
                    started_at[index],
                )
                continue
            assert query is not None
            prepared.append((index, request_id, query))

        if not prepared:
            return
        try:
            result_batches = backend.batch_search(
                [query for _, _, query in prepared],
                top_k=self.config.search_top_k,
            )
            if len(result_batches) != len(prepared):
                raise RuntimeError(
                    "search backend returned a different number of result "
                    "groups than valid queries"
                )
            for (index, request_id, _), search_results in zip(
                prepared,
                result_batches,
            ):
                results[index] = self._batch_result(
                    request_id,
                    self._wrap(format_search_results(search_results)),
                    "ok",
                    started_at[index],
                )
        except BaseException as exc:
            error = f"Error: search backend failed ({type(exc).__name__}: {exc})"
            for index, request_id, _ in prepared:
                results[index] = self._batch_error(
                    request_id,
                    error,
                    started_at[index],
                )

    def _prepare_search_query(
        self,
        raw_text: object,
        tag: str,
    ) -> tuple[str | None, str | None]:
        input_error = self._validate_raw_text(raw_text)
        if input_error:
            return None, input_error
        query = extract_tagged_content(raw_text, tag)
        if query is None:
            return None, f"Error: no <{tag}> query detected."
        if not query:
            return None, "Error: empty query."
        if len(query) > self.config.search_query_max_chars:
            return (
                None,
                "Error: search query exceeds the configured length limit.",
            )
        return query, None

    def health(self) -> dict[str, object]:
        import torch

        cuda_device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        return {
            "status": "ok",
            "python_execution_result_version": PYTHON_EXECUTION_RESULT_VERSION,
            "gpu": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "physical_gpu_uuid": os.environ.get("TRACE_OPSD_PHYSICAL_GPU_UUID", ""),
                "device_count": cuda_device_count,
                "device_name": (
                    torch.cuda.get_device_name(0) if cuda_device_count == 1 else None
                ),
            },
            "search": self.search_backend.health(),
            "python_executor": self.python_executor.capabilities(),
        }

    def _validate_raw_text(self, raw_text: object) -> str | None:
        if not isinstance(raw_text, str):
            return "Error: raw_text must be a string."
        if len(raw_text) > self.config.raw_text_max_chars:
            return "Error: raw_text exceeds the configured length limit."
        return None

    def _wrap(self, content: object) -> str:
        return wrap_result(content, max_chars=self.config.result_max_chars)

    @staticmethod
    def _result_status(result: str) -> str:
        return "error" if result.startswith("<result>\nError") else "ok"

    def _batch_error(
        self,
        request_id: object,
        message: object,
        started_at: float,
    ) -> dict[str, object]:
        return self._batch_result(
            request_id,
            self._wrap(message),
            "error",
            started_at,
        )

    @staticmethod
    def _batch_result(
        request_id: object,
        result: str,
        status: str,
        started_at: float,
        *,
        error_category: str | None = None,
        error_origin: str | None = None,
        execution_result_version: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": request_id,
            "result": result,
            "status": status,
            "latency_ms": round(max(0.0, (perf_counter() - started_at) * 1000), 3),
        }
        if error_category is not None:
            payload["error_category"] = error_category
        if execution_result_version is not None:
            payload["execution_result_version"] = execution_result_version
            payload["error_origin"] = error_origin
        return payload


_DEFAULT_RUNTIME: ToolRuntime | None = None
_DEFAULT_RUNTIME_LOCK = RLock()


def initialize_tools(config: ToolConfig) -> ToolRuntime:
    """Initialize the process-global runtime once, without partial publication."""

    global _DEFAULT_RUNTIME
    with _DEFAULT_RUNTIME_LOCK:
        if _DEFAULT_RUNTIME is not None:
            if _DEFAULT_RUNTIME.config != config:
                raise RuntimeError(
                    "tools are already initialized with a different configuration"
                )
            return _DEFAULT_RUNTIME
        runtime = ToolRuntime.from_config(config)
        _DEFAULT_RUNTIME = runtime
        return runtime


def _call_default(method: str, raw_text: object) -> str:
    with _DEFAULT_RUNTIME_LOCK:
        runtime = _DEFAULT_RUNTIME
    if runtime is None:
        return wrap_result("Error: tool runtime is not initialized.")
    try:
        return getattr(runtime, method)(raw_text)
    except BaseException as exc:
        return wrap_result(
            f"Error: tool runtime failed ({type(exc).__name__}: {exc})",
            max_chars=runtime.config.result_max_chars,
        )


def call_tool(tool_name: object, raw_text: object) -> str:
    with _DEFAULT_RUNTIME_LOCK:
        runtime = _DEFAULT_RUNTIME
    if runtime is None:
        return wrap_result("Error: tool runtime is not initialized.")
    return runtime.call_tool(tool_name, raw_text)


def python_executor_tool(raw_text: object) -> str:
    return _call_default("python_executor_tool", raw_text)


def search_tool(raw_text: object) -> str:
    return _call_default("search_tool", raw_text)
