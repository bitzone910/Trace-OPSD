"""Stable search abstractions over the existing dense retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
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

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 3,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in input order.

        Custom backends only need to implement :meth:`search`; implementations
        with a vectorized path can override this method.
        """

        return [self.search(query, top_k=top_k) for query in queries]

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
        return self._normalize_hits(hits)

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: int = 3,
    ) -> list[list[SearchResult]]:
        query_list = list(queries)
        if not query_list:
            return []

        retriever_batch_search = getattr(self.retriever, "batch_search", None)
        if not callable(retriever_batch_search):
            return super().batch_search(query_list, top_k=top_k)

        # Encoder and index/model state are protected for the whole native
        # batch, rather than acquiring the lock separately for each query.
        with self._lock:
            hit_batches = list(retriever_batch_search(query_list, topk=top_k))
        if len(hit_batches) != len(query_list):
            raise RuntimeError(
                "retriever batch_search returned a different number of result "
                "groups than input queries"
            )
        return [self._normalize_hits(hits) for hits in hit_batches]

    @staticmethod
    def _normalize_hits(hits: Sequence[dict[str, Any]]) -> list[SearchResult]:
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
