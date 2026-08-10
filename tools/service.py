"""Unified FastAPI service for all three rollout tools."""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from shared.config import DEFAULT_MODEL

from .config import ToolConfig
from .runtime import ToolRuntime, initialize_tools


class ToolRequest(BaseModel):
    raw_text: str


class ToolResponse(BaseModel):
    result: str


def create_app(runtime: ToolRuntime) -> FastAPI:
    """Create an app around an already initialized runtime."""

    app = FastAPI(title="ToolElite Tool Backend", version="1.0.0")
    app.state.tool_runtime = runtime

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.post("/tools/python", response_model=ToolResponse)
    def python_tool(request: ToolRequest) -> dict[str, str]:
        return {"result": runtime.python_executor_tool(request.raw_text)}

    @app.post("/tools/search_knowledge", response_model=ToolResponse)
    def search_knowledge(request: ToolRequest) -> dict[str, str]:
        return {"result": runtime.search_knowledge_tool(request.raw_text)}

    @app.post("/tools/search_theory", response_model=ToolResponse)
    def search_theory(request: ToolRequest) -> dict[str, str]:
        return {"result": runtime.search_theory_tool(request.raw_text)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch all ToolElite tools.")
    parser.add_argument("--general_index_path", required=True)
    parser.add_argument("--general_corpus_path", required=True)
    parser.add_argument("--theorem_index_dir", required=True)
    parser.add_argument("--theorem_corpus_path", required=True)
    parser.add_argument("--retriever_model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--corpus_num_proc", type=int, default=4)
    parser.add_argument("--slogan_topk", type=int, default=20)
    parser.add_argument("--search_topk", type=int, default=3)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        theorem_index_dir=args.theorem_index_dir,
        theorem_corpus_path=args.theorem_corpus_path,
        model_path=args.retriever_model,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        faiss_gpu=args.faiss_gpu,
        retrieval_batch_size=args.retrieval_batch_size,
        corpus_num_proc=args.corpus_num_proc,
        theorem_slogan_topk=args.slogan_topk,
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
