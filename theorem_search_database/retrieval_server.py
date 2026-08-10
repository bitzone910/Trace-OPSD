"""FastAPI service for the two-stage theorem search database."""

from __future__ import annotations

import argparse
from typing import Any

import faiss
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from shared.config import DEFAULT_MODEL
from shared.validation import omit_scores
from theorem_search_database.retriever import TheoremRetriever


class QueryRequest(BaseModel):
    queries: list[str] = Field(min_length=1)
    topk: int | None = Field(default=None, gt=0)
    return_scores: bool = True


app = FastAPI(title="Theorem Search Database", version="1.0.0")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")
    return {
        "status": "ok",
        "database": "theorem",
        "documents": len(retriever.corpus),
        "unique_slogans": int(retriever.index.ntotal),
        "slogan_topk": retriever.slogan_topk,
        "device": str(retriever.encoder.device),
        "embedding_device": str(retriever.encoder.device),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_device": "gpu" if retriever.faiss_gpu else "cpu",
        "faiss_gpus": retriever.faiss_gpu_count,
    }


@app.post("/retrieve")
def retrieve(request_body: QueryRequest, request: Request) -> dict[str, Any]:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized")
    topk = (
        request_body.topk
        if request_body.topk is not None
        else request.app.state.default_topk
    )
    try:
        results = retriever.batch_search(request_body.queries, topk=topk)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "result": results if request_body.return_scores else omit_scores(results)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Theorem Search Database.")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--slogan_topk", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--corpus_num_proc", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be greater than zero")
    if args.slogan_topk <= 0:
        raise ValueError("slogan_topk must be greater than zero")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    app.state.retriever = TheoremRetriever(
        args.index_dir,
        args.corpus_path,
        model_path=args.retriever_model,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        faiss_gpu=args.faiss_gpu,
        batch_size=args.batch_size,
        corpus_num_proc=args.corpus_num_proc,
        slogan_topk=args.slogan_topk,
    )
    app.state.default_topk = args.topk
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
