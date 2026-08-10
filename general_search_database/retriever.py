"""Dense query-to-contents retrieval for the general search database."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
from tqdm import tqdm

from shared.config import DEFAULT_MODEL
from shared.corpus import load_corpus as load_shared_corpus
from shared.encoder import E5Encoder
from shared.faiss_utils import move_index_to_gpu
from shared.validation import (
    effective_topk,
    validate_embedding_dimension,
    validate_query,
)


def load_corpus(corpus_path: str | Path, *, num_proc: int = 4):
    return load_shared_corpus(
        corpus_path,
        num_proc=num_proc,
        require_slogan=False,
    )


class DenseRetriever:
    """Load a FAISS index and retrieve original id/contents rows."""

    def __init__(
        self,
        index_path: str | Path,
        corpus_path: str | Path,
        *,
        model_path: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 256,
        use_fp16: bool = False,
        faiss_gpu: bool = True,
        batch_size: int = 128,
        corpus_num_proc: int = 4,
        encoder: E5Encoder | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        resolved_index_path = Path(index_path).expanduser().resolve()
        if not resolved_index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index file does not exist: {resolved_index_path}"
            )
        self.index = faiss.read_index(str(resolved_index_path))
        self.faiss_gpu = faiss_gpu
        self.faiss_gpu_count = 0
        if faiss_gpu:
            self.index, self.faiss_gpu_count = move_index_to_gpu(self.index)
        self._index_lock = RLock()

        self.corpus = load_corpus(corpus_path, num_proc=corpus_num_proc)
        if int(self.index.ntotal) != len(self.corpus):
            raise ValueError(
                "Index/corpus size mismatch: "
                f"index has {int(self.index.ntotal)} vectors but corpus has "
                f"{len(self.corpus)} rows. Rebuild the index from this exact "
                "corpus file and preserve JSONL row order."
            )

        self.encoder = encoder or E5Encoder(
            model_path,
            device=device,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        validate_embedding_dimension(
            int(self.index.d),
            self.encoder.embedding_dimension,
            self.encoder.model_path,
        )
        self.batch_size = batch_size

    def _format_hits(
        self,
        indices: Sequence[int],
        scores: Sequence[float],
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for doc_index, score in zip(indices, scores):
            if int(doc_index) < 0:
                continue
            document = self.corpus[int(doc_index)]
            hits.append(
                {
                    "id": document["id"],
                    "contents": document["contents"],
                    "score": float(score),
                }
            )
        return hits

    def search(self, query: str, topk: int = 3) -> list[dict[str, Any]]:
        validate_query(query)
        search_topk = effective_topk(topk, int(self.index.ntotal))
        query_embedding = self.encoder.encode(query, is_query=True)
        with self._index_lock:
            scores, indices = self.index.search(query_embedding, search_topk)
        return self._format_hits(indices[0].tolist(), scores[0].tolist())

    def batch_search(
        self,
        queries: Sequence[str],
        topk: int = 3,
    ) -> list[list[dict[str, Any]]]:
        query_list = list(queries)
        if not query_list:
            raise ValueError("queries must contain at least one query")
        for query in query_list:
            validate_query(query)
        search_topk = effective_topk(topk, int(self.index.ntotal))

        all_results: list[list[dict[str, Any]]] = []
        batch_starts = range(0, len(query_list), self.batch_size)
        for start in tqdm(
            batch_starts,
            desc="Retrieval",
            disable=len(query_list) == 1,
        ):
            query_batch = query_list[start : start + self.batch_size]
            embeddings = self.encoder.encode(query_batch, is_query=True)
            with self._index_lock:
                batch_scores, batch_indices = self.index.search(
                    embeddings,
                    search_topk,
                )
            for indices, scores in zip(batch_indices, batch_scores):
                all_results.append(
                    self._format_hits(indices.tolist(), scores.tolist())
                )
        return all_results
