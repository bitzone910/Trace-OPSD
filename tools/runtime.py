"""Tool registry, lifecycle, and total public call boundaries."""

from __future__ import annotations

from threading import RLock
from typing import Callable

from .common import extract_tagged_content, wrap_result
from .config import ToolConfig
from .python_executor import PythonExecutor, find_dangerous_operation
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
        knowledge_backend: SearchBackend,
        theory_backend: SearchBackend,
        python_executor: PythonExecutor | None = None,
    ) -> None:
        self.config = config
        self.knowledge_backend = knowledge_backend
        self.theory_backend = theory_backend
        self.python_executor = python_executor or PythonExecutor(config)
        self._registry: dict[str, Callable[[object], str]] = {
            "python": self.python_executor_tool,
            "search_knowledge": self.search_knowledge_tool,
            "search_theory": self.search_theory_tool,
        }

    @classmethod
    def from_config(cls, config: ToolConfig) -> "ToolRuntime":
        """Load one encoder and both existing retrievers exactly once."""

        from general_search_database.retriever import DenseRetriever
        from shared.encoder import E5Encoder
        from theorem_search_database.retriever import TheoremRetriever

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
        theorem = TheoremRetriever(
            config.theorem_index_dir,
            config.theorem_corpus_path,
            encoder=encoder,
            faiss_gpu=config.faiss_gpu,
            batch_size=config.retrieval_batch_size,
            corpus_num_proc=config.corpus_num_proc,
            slogan_topk=config.theorem_slogan_topk,
        )
        return cls(
            config,
            knowledge_backend=RetrieverSearchBackend(
                general, database="general"
            ),
            theory_backend=RetrieverSearchBackend(theorem, database="theorem"),
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

    def python_executor_tool(self, raw_text: object) -> str:
        try:
            input_error = self._validate_raw_text(raw_text)
            if input_error:
                return self._wrap(input_error)
            code = extract_tagged_content(raw_text, "python")
            if code is None:
                return self._wrap("Error: no <python> code block detected.")
            if not code:
                return self._wrap("Error: empty code block.")
            if len(code) > self.config.python_code_max_chars:
                return self._wrap(
                    "Error: Python code exceeds the configured length limit."
                )
            danger = find_dangerous_operation(code)
            if danger:
                return self._wrap(
                    f"Error: restricted operation '{danger}' is not allowed."
                )
            return self._wrap(self.python_executor.execute(code))
        except BaseException as exc:
            return self._wrap(
                f"Error: Python Executor failed ({type(exc).__name__}: {exc})"
            )

    def search_knowledge_tool(self, raw_text: object) -> str:
        return self._search_tool(
            raw_text,
            tag="search_knowledge",
            backend=self.knowledge_backend,
        )

    def search_theory_tool(self, raw_text: object) -> str:
        return self._search_tool(
            raw_text,
            tag="search_theory",
            backend=self.theory_backend,
        )

    def _search_tool(
        self,
        raw_text: object,
        *,
        tag: str,
        backend: SearchBackend,
    ) -> str:
        try:
            input_error = self._validate_raw_text(raw_text)
            if input_error:
                return self._wrap(input_error)
            query = extract_tagged_content(raw_text, tag)
            if query is None:
                return self._wrap(f"Error: no <{tag}> query detected.")
            if not query:
                return self._wrap("Error: empty query.")
            if len(query) > self.config.search_query_max_chars:
                return self._wrap(
                    "Error: search query exceeds the configured length limit."
                )
            results = backend.search(query, top_k=self.config.search_top_k)
            return self._wrap(format_search_results(results))
        except BaseException as exc:
            return self._wrap(
                f"Error: search backend failed ({type(exc).__name__}: {exc})"
            )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "knowledge": self.knowledge_backend.health(),
            "theory": self.theory_backend.health(),
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


def search_knowledge_tool(raw_text: object) -> str:
    return _call_default("search_knowledge_tool", raw_text)


def search_theory_tool(raw_text: object) -> str:
    return _call_default("search_theory_tool", raw_text)
