"""FAISS construction and GPU transfer helpers."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import torch


SUPPORTED_INDEX_TYPES = ("Flat", "HNSW64")


def require_faiss_gpu() -> int:
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


def create_faiss_index(
    dimension: int,
    *,
    faiss_type: str,
    faiss_gpu: bool,
) -> Any:
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")
    if faiss_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(
            f"Unsupported faiss_type '{faiss_type}'. "
            f"Choose one of: {', '.join(SUPPORTED_INDEX_TYPES)}"
        )
    if faiss_gpu and faiss_type != "Flat":
        raise ValueError("GPU FAISS is supported only with faiss_type='Flat'")

    index = faiss.index_factory(
        dimension,
        faiss_type,
        faiss.METRIC_INNER_PRODUCT,
    )
    if faiss_gpu:
        require_faiss_gpu()
        options = faiss.GpuMultipleClonerOptions()
        options.useFloat16 = True
        options.shard = True
        index = faiss.index_cpu_to_all_gpus(index, options)
    return index


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    faiss_type: str,
    faiss_gpu: bool,
) -> Any:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
    index = create_faiss_index(
        int(embeddings.shape[1]),
        faiss_type=faiss_type,
        faiss_gpu=faiss_gpu,
    )
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    return faiss.index_gpu_to_cpu(index) if faiss_gpu else index


def move_index_to_gpu(index: Any) -> tuple[Any, int]:
    gpu_count = require_faiss_gpu()
    try:
        options = faiss.GpuMultipleClonerOptions()
        options.useFloat16 = True
        options.shard = True
        return faiss.index_cpu_to_all_gpus(index, options), gpu_count
    except Exception as exc:
        raise RuntimeError(
            "Could not move the FAISS index to GPU. Use a Flat index, or "
            "run with GPU FAISS disabled."
        ) from exc
