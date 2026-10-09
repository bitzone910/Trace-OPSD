"""Unified FastAPI service for Python and General Search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import Body, FastAPI
from pydantic import BaseModel

from trace_opsd.retrieval.config import DEFAULT_MODEL

from .common import wrap_result
from .config import ToolConfig
from .python_executor import PYTHON_EXECUTION_RESULT_VERSION
from .runtime import ToolRuntime, initialize_tools

TOOL_SERVICE_VERSION = "2.0.0"


class ToolRequest(BaseModel):
    raw_text: str


class ToolResponse(BaseModel):
    result: str
    status: str | None = None
    error_category: str | None = None
    execution_result_version: str | None = None
    error_origin: str | None = None


class BatchToolResponse(BaseModel):
    request_id: Any = None
    result: str
    status: str
    latency_ms: float
    error_category: str | None = None
    execution_result_version: str | None = None
    error_origin: str | None = None


class BatchResponse(BaseModel):
    results: list[BatchToolResponse]


def create_app(runtime: ToolRuntime) -> FastAPI:
    """Create an app around an already initialized runtime."""

    app = FastAPI(title="Trace-OPSD Tool Backend", version=TOOL_SERVICE_VERSION)
    app.state.tool_runtime = runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        evaluation_config = {
            key: str(Path(value).resolve())
            if key in {"general_index_path", "general_corpus_path", "model_path"}
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in asdict(runtime.config).items()
            if key not in {"executor_uid", "executor_gid"}
        }
        return {
            **runtime.health(),
            "service_version": TOOL_SERVICE_VERSION,
            "batch_endpoint": "/tools/batch",
            "evaluation_config": evaluation_config,
        }

    @app.post("/tools/python", response_model=ToolResponse)
    def python_tool(request: ToolRequest) -> dict[str, Any]:
        outcome = runtime.python_executor_result(request.raw_text)
        return {
            "result": outcome.observation,
            "status": outcome.status,
            "error_category": outcome.error_category,
            "execution_result_version": outcome.execution_result_version,
            "error_origin": outcome.error_origin,
        }

    @app.post("/tools/search", response_model=ToolResponse)
    def search(request: ToolRequest) -> dict[str, str]:
        return {"result": runtime.search_tool(request.raw_text)}

    @app.post("/tools/batch", response_model=BatchResponse)
    def batch_tools(payload: Any = Body(...)) -> dict[str, object]:
        started_at = perf_counter()
        try:
            calls = payload.get("calls") if isinstance(payload, dict) else None
            return {"results": runtime.batch_call(calls)}
        except BaseException as exc:
            result = wrap_result(
                f"Error: batch dispatch failed ({type(exc).__name__}: {exc})",
                max_chars=runtime.config.result_max_chars,
            )
            return {
                "results": [
                    {
                        "request_id": None,
                        "result": result,
                        "status": "error",
                        "error_category": "sandbox_error",
                        "error_origin": "sandbox",
                        "execution_result_version": PYTHON_EXECUTION_RESULT_VERSION,
                        "latency_ms": round(
                            max(0.0, (perf_counter() - started_at) * 1000),
                            3,
                        ),
                    }
                ]
            }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch all Trace-OPSD tools.")
    parser.add_argument("--general_index_path", required=True)
    parser.add_argument("--general_corpus_path", required=True)
    parser.add_argument("--retriever_model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--corpus_num_proc", type=int, default=4)
    parser.add_argument("--search_topk", type=int, default=3)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--result_max_chars", type=int, default=2000)
    parser.add_argument("--raw_text_max_chars", type=int, default=100_000)
    parser.add_argument("--python_code_max_chars", type=int, default=20_000)
    parser.add_argument("--search_query_max_chars", type=int, default=2_000)
    parser.add_argument("--python_timeout", type=float, default=10.0)
    parser.add_argument("--python_cpu_seconds", type=int, default=10)
    parser.add_argument("--python_memory_mb", type=int, default=512)
    parser.add_argument("--python_output_bytes", type=int, default=64 * 1024)
    parser.add_argument("--python_nofile", type=int, default=64)
    parser.add_argument("--python_max_concurrency", type=int, default=8)
    parser.add_argument("--python_queue_timeout", type=float, default=30.0)
    parser.add_argument("--executor_uid", type=int)
    parser.add_argument("--executor_gid", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ToolConfig:
    return ToolConfig(
        general_index_path=args.general_index_path,
        general_corpus_path=args.general_corpus_path,
        model_path=args.retriever_model,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        faiss_gpu=args.faiss_gpu,
        retrieval_batch_size=args.retrieval_batch_size,
        corpus_num_proc=args.corpus_num_proc,
        search_top_k=args.search_topk,
        result_max_chars=args.result_max_chars,
        raw_text_max_chars=args.raw_text_max_chars,
        python_code_max_chars=args.python_code_max_chars,
        search_query_max_chars=args.search_query_max_chars,
        python_timeout_seconds=args.python_timeout,
        python_cpu_seconds=args.python_cpu_seconds,
        python_memory_limit_mb=args.python_memory_mb,
        python_output_limit_bytes=args.python_output_bytes,
        python_nofile_limit=args.python_nofile,
        python_max_concurrency=args.python_max_concurrency,
        python_queue_timeout_seconds=args.python_queue_timeout,
        executor_uid=args.executor_uid,
        executor_gid=args.executor_gid,
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    runtime = initialize_tools(config_from_args(args))

    import uvicorn

    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
