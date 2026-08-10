"""Configuration for the importable and HTTP tool runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.config import DEFAULT_MODEL


@dataclass(frozen=True)
class ToolConfig:
    """All paths and bounded-resource defaults needed by a tool runtime."""

    general_index_path: str | Path
    general_corpus_path: str | Path
    theorem_index_dir: str | Path
    theorem_corpus_path: str | Path
    model_path: str = DEFAULT_MODEL
    device: str = "auto"
    max_length: int = 256
    use_fp16: bool = False
    faiss_gpu: bool = True
    retrieval_batch_size: int = 128
    corpus_num_proc: int = 4
    theorem_slogan_topk: int = 20
    search_top_k: int = 3
    result_max_chars: int = 2000
    raw_text_max_chars: int = 100_000
    python_code_max_chars: int = 20_000
    search_query_max_chars: int = 2_000
    python_timeout_seconds: float = 10.0
    python_cpu_seconds: int = 10
    python_memory_limit_mb: int = 512
    python_output_limit_bytes: int = 64 * 1024
    python_nofile_limit: int = 64
    python_max_concurrency: int = 8
    python_queue_timeout_seconds: float = 30.0
    executor_uid: int | None = None
    executor_gid: int | None = None

    def __post_init__(self) -> None:
        positive_ints = {
            "max_length": self.max_length,
            "retrieval_batch_size": self.retrieval_batch_size,
            "corpus_num_proc": self.corpus_num_proc,
            "theorem_slogan_topk": self.theorem_slogan_topk,
            "search_top_k": self.search_top_k,
            "result_max_chars": self.result_max_chars,
            "raw_text_max_chars": self.raw_text_max_chars,
            "python_code_max_chars": self.python_code_max_chars,
            "search_query_max_chars": self.search_query_max_chars,
            "python_cpu_seconds": self.python_cpu_seconds,
            "python_memory_limit_mb": self.python_memory_limit_mb,
            "python_output_limit_bytes": self.python_output_limit_bytes,
            "python_nofile_limit": self.python_nofile_limit,
            "python_max_concurrency": self.python_max_concurrency,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "python_timeout_seconds": self.python_timeout_seconds,
            "python_queue_timeout_seconds": self.python_queue_timeout_seconds,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a positive number")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if (self.executor_uid is None) != (self.executor_gid is None):
            raise ValueError("executor_uid and executor_gid must be set together")
        for name, value in {
            "executor_uid": self.executor_uid,
            "executor_gid": self.executor_gid,
        }.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
