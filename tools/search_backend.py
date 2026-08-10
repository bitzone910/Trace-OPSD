"""Stable search abstractions over the existing dense retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    content: str
    score: float = 0.0


class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return results in descending relevance order."""

        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}


class RetrieverSearchBackend(SearchBackend):
    """Normalize an existing General or Theorem retriever."""

    def __init__(self, retriever: Any, *, database: str) -> None:
        self.retriever = retriever
        self.database = database
        self._lock = RLock()

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        # The adapter lock protects third-party/model state even if a different
        # retriever implementation is injected later.
        with self._lock:
            hits = self.retriever.search(query, topk=top_k)
        return [
            SearchResult(
                document_id=str(hit["id"]),
                content=str(hit["contents"]),
                score=float(hit.get("score", 0.0)),
            )
            for hit in hits
        ]

    def health(self) -> dict[str, Any]:
        retriever = self.retriever
        data: dict[str, Any] = {
            "status": "ok",
            "database": self.database,
            "device": str(retriever.encoder.device),
            "faiss_device": "gpu" if retriever.faiss_gpu else "cpu",
            "faiss_gpus": int(retriever.faiss_gpu_count),
        }
        if self.database == "general":
            data["documents"] = int(retriever.index.ntotal)
        else:
            data.update(
                documents=len(retriever.corpus),
                unique_slogans=int(retriever.index.ntotal),
                slogan_topk=int(retriever.slogan_topk),
            )
        return data


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant results found."
    return "\n\n".join(
        f"[{index}] {result.document_id}\n{result.content}"
        for index, result in enumerate(results, 1)
    )
