# Adapted from Search-R1 under Apache-2.0.
# Modified for standalone dense retrieval, device selection, and validation.
"""Dense E5 + FAISS retrieval primitives extracted from Search-R1.

The retrieval algorithm intentionally follows Search-R1: E5 query/passage
prefixes, mean pooling, L2-normalized embeddings, and inner-product search.
This file contains no LLM training or rollout code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "intfloat/e5-base-v2"


def require_faiss_gpu() -> int:
    """Validate the CUDA-enabled FAISS runtime and return visible GPU count."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU FAISS requires an available CUDA device, but "
            "torch.cuda.is_available() is false."
        )
    if not hasattr(faiss, "index_cpu_to_all_gpus") or not hasattr(
        faiss, "get_num_gpus"
    ):
        raise RuntimeError(
            "GPU FAISS APIs are unavailable. Install "
            "faiss-gpu-cuvs=1.12.0 with Conda; do not install faiss-cpu."
        )

    gpu_count = int(faiss.get_num_gpus())
    if gpu_count <= 0:
        raise RuntimeError(
            "FAISS cannot see any CUDA GPUs. Check CUDA_VISIBLE_DEVICES and "
            "the NVIDIA driver."
        )
    return gpu_count


def load_corpus(corpus_path: str | Path, *, num_proc: int = 4) -> Dataset:
    """Load and validate a Search-R1-style JSONL corpus.

    FAISS vector ids are positional row ids, so callers must keep this dataset
    in exactly the same order while building and querying an index.
    """

    path = Path(corpus_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    corpus = load_dataset(
        "json",
        data_files=str(path),
        split="train",
        num_proc=max(1, num_proc),
    )
    if len(corpus) == 0:
        raise ValueError(f"Corpus is empty: {path}")

    missing_columns = {"id", "contents"} - set(corpus.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Corpus is missing required field(s): {missing}")

    for row_index, row in enumerate(corpus):
        if row["id"] is None:
            raise ValueError(f"Corpus row {row_index + 1} has a null 'id'")
        if not isinstance(row["contents"], str):
            raise ValueError(
                f"Corpus row {row_index + 1} has a non-string 'contents'"
            )

    return corpus


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and otherwise to CPU."""

    normalized = device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return resolved


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool non-padding tokens, matching Search-R1's E5 behavior."""

    masked_hidden = last_hidden_state.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    token_counts = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return masked_hidden.sum(dim=1) / token_counts


class E5Encoder:
    """Hugging Face encoder with Search-R1-compatible E5 preprocessing."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        max_length: int = 256,
        use_fp16: bool = False,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        self.model_path = model_path
        self.device = resolve_device(device)
        self.max_length = max_length
        self.use_fp16 = use_fp16

        if use_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 encoding is only supported on a CUDA device")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)
        if use_fp16:
            self.model.half()

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Model does not expose config.hidden_size: {model_path}"
            )
        self.embedding_dimension = int(hidden_size)

    @torch.inference_mode()
    def encode(
        self, texts: str | Sequence[str], *, is_query: bool
    ) -> np.ndarray:
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)
        if not text_list:
            raise ValueError("At least one text is required for encoding")
        if any(not isinstance(text, str) for text in text_list):
            raise TypeError("All texts must be strings")

        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [f"{prefix}{text}" for text in text_list]
        inputs = self.tokenizer(
            prefixed_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device_inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        output = self.model(**device_inputs, return_dict=True)
        embeddings = mean_pool(
            output.last_hidden_state,
            device_inputs["attention_mask"],
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        array = embeddings.detach().float().cpu().numpy()
        array = np.asarray(array, dtype=np.float32, order="C")

        del device_inputs, output, embeddings
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return array


class DenseRetriever:
    """Load a FAISS index and retrieve original ``id``/``contents`` rows."""

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
            self.faiss_gpu_count = require_faiss_gpu()
            try:
                options = faiss.GpuMultipleClonerOptions()
                options.useFloat16 = True
                options.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, options)
            except Exception as exc:
                raise RuntimeError(
                    "Could not move the FAISS index to GPU. GPU FAISS supports "
                    "only selected index types. Use a Flat index, or pass "
                    "faiss_gpu=False for a CPU-only index."
                ) from exc

        self.corpus = load_corpus(corpus_path, num_proc=corpus_num_proc)
        if int(self.index.ntotal) != len(self.corpus):
            raise ValueError(
                "Index/corpus size mismatch: "
                f"index has {int(self.index.ntotal)} vectors but corpus has "
                f"{len(self.corpus)} rows. Rebuild the index from this exact "
                "corpus file and preserve JSONL row order."
            )

        self.encoder = E5Encoder(
            model_path,
            device=device,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        if int(self.index.d) != self.encoder.embedding_dimension:
            raise ValueError(
                "Model/index dimension mismatch: "
                f"index dimension is {int(self.index.d)}, but model "
                f"'{model_path}' outputs {self.encoder.embedding_dimension}. "
                "Start the server with the same model used to build the index."
            )

        self.batch_size = batch_size

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")

    def _effective_topk(self, topk: int) -> int:
        if isinstance(topk, bool) or not isinstance(topk, int):
            raise TypeError("topk must be an integer")
        if topk <= 0:
            raise ValueError("topk must be greater than zero")
        return min(topk, int(self.index.ntotal))

    def _format_hits(
        self, indices: Sequence[int], scores: Sequence[float]
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
        """Retrieve top-k documents for one query."""

        self._validate_query(query)
        effective_topk = self._effective_topk(topk)
        query_embedding = self.encoder.encode(query, is_query=True)
        scores, indices = self.index.search(query_embedding, effective_topk)
        return self._format_hits(indices[0].tolist(), scores[0].tolist())

    def batch_search(
        self, queries: Sequence[str], topk: int = 3
    ) -> list[list[dict[str, Any]]]:
        """Retrieve top-k documents for a batch of queries."""

        query_list = list(queries)
        if not query_list:
            raise ValueError("queries must contain at least one query")
        for query in query_list:
            self._validate_query(query)
        effective_topk = self._effective_topk(topk)

        all_results: list[list[dict[str, Any]]] = []
        batch_starts = range(0, len(query_list), self.batch_size)
        for start in tqdm(batch_starts, desc="Retrieval", disable=len(query_list) == 1):
            query_batch = query_list[start : start + self.batch_size]
            embeddings = self.encoder.encode(query_batch, is_query=True)
            batch_scores, batch_indices = self.index.search(
                embeddings, effective_topk
            )
            for indices, scores in zip(batch_indices, batch_scores):
                all_results.append(
                    self._format_hits(indices.tolist(), scores.tolist())
                )
        return all_results
