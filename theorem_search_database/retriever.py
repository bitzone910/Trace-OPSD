"""Two-stage unique-slogan recall and theorem-content reranking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from shared.config import DEFAULT_MODEL
from shared.corpus import load_corpus
from shared.encoder import E5Encoder
from shared.faiss_utils import move_index_to_gpu
from shared.validation import (
    effective_topk,
    validate_embedding_dimension,
    validate_query,
)
from theorem_search_database.artifacts import (
    CONTENT_EMBEDDINGS_FILE,
    CONTENT_ROW_IDS_FILE,
    SLOGAN_INDEX_FILE,
    SLOGAN_OFFSETS_FILE,
    load_array,
    read_manifest,
    validate_bundle_arrays,
)
from theorem_search_database.core import (
    expand_slogan_groups,
    rank_candidate_rows,
)


class TheoremRetriever:
    """Retrieve slogans, expand their contents, then rerank the contents."""

    def __init__(
        self,
        index_dir: str | Path,
        corpus_path: str | Path,
        *,
        model_path: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 256,
        use_fp16: bool = False,
        faiss_gpu: bool = True,
        batch_size: int = 128,
        corpus_num_proc: int = 4,
        slogan_topk: int = 20,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if slogan_topk <= 0:
            raise ValueError("slogan_topk must be greater than zero")

        self.index_dir = Path(index_dir).expanduser().resolve()
        if not self.index_dir.is_dir():
            raise FileNotFoundError(
                f"Theorem index bundle does not exist: {self.index_dir}"
            )
        self.manifest = read_manifest(self.index_dir)
        index_path = self.index_dir / SLOGAN_INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(f"Slogan FAISS index does not exist: {index_path}")
        self.index = faiss.read_index(str(index_path))
        self.faiss_gpu = faiss_gpu
        self.faiss_gpu_count = 0
        if faiss_gpu:
            self.index, self.faiss_gpu_count = move_index_to_gpu(self.index)

        self.corpus = load_corpus(
            corpus_path,
            num_proc=corpus_num_proc,
            require_slogan=True,
        )
        self.slogan_offsets = load_array(
            self.index_dir / SLOGAN_OFFSETS_FILE
        )
        self.content_row_ids = load_array(
            self.index_dir / CONTENT_ROW_IDS_FILE
        )
        self.content_embeddings = load_array(
            self.index_dir / CONTENT_EMBEDDINGS_FILE
        )

        self._validate_bundle()
        self.encoder = E5Encoder(
            model_path,
            device=device,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        validate_embedding_dimension(
            int(self.index.d),
            self.encoder.embedding_dimension,
            model_path,
        )
        self.batch_size = batch_size
        self.slogan_topk = slogan_topk

    def _validate_bundle(self) -> None:
        validate_bundle_arrays(
            self.manifest,
            index_total=int(self.index.ntotal),
            index_dimension=int(self.index.d),
            corpus_rows=len(self.corpus),
            slogan_offsets=self.slogan_offsets,
            content_row_ids=self.content_row_ids,
            content_embeddings=self.content_embeddings,
        )

    def _search_from_embedding(
        self,
        query_embedding: np.ndarray,
        slogan_group_ids: np.ndarray,
        *,
        topk: int,
    ) -> list[dict[str, Any]]:
        candidate_row_ids = expand_slogan_groups(
            slogan_group_ids,
            self.slogan_offsets,
            self.content_row_ids,
        )
        ranked_row_ids, scores = rank_candidate_rows(
            query_embedding,
            self.content_embeddings,
            candidate_row_ids,
            topk=topk,
        )
        hits: list[dict[str, Any]] = []
        for row_id, score in zip(ranked_row_ids, scores):
            row = self.corpus[int(row_id)]
            hits.append(
                {
                    "id": row["id"],
                    "contents": row["contents"],
                    "score": float(score),
                }
            )
        return hits

    def search(self, query: str, topk: int = 3) -> list[dict[str, Any]]:
        return self.batch_search([query], topk=topk)[0]

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
        # Validate even though the final candidate count is known only later.
        effective_topk(topk, len(self.corpus))
        first_stage_topk = effective_topk(
            self.slogan_topk,
            int(self.index.ntotal),
        )

        all_results: list[list[dict[str, Any]]] = []
        for start in tqdm(
            range(0, len(query_list), self.batch_size),
            desc="Theorem retrieval",
            disable=len(query_list) == 1,
        ):
            query_batch = query_list[start : start + self.batch_size]
            embeddings = self.encoder.encode(query_batch, is_query=True)
            _, group_ids = self.index.search(embeddings, first_stage_topk)
            for embedding, query_group_ids in zip(embeddings, group_ids):
                all_results.append(
                    self._search_from_embedding(
                        embedding,
                        query_group_ids,
                        topk=topk,
                    )
                )
        return all_results
