# Adapted from Search-R1 under Apache-2.0.
# Modified to provide a dense-only standalone CLI and safer output handling.
"""Build Search-R1-compatible E5 FAISS indexes from a JSONL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from tqdm import tqdm

from retriever import DEFAULT_MODEL, E5Encoder, load_corpus, require_faiss_gpu


SUPPORTED_INDEX_TYPES = ("Flat", "HNSW64")


def encode_corpus(
    corpus,
    encoder: E5Encoder,
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode corpus rows in JSONL order, preserving FAISS row ids."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    encoded_batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(corpus), batch_size), desc="Encoding corpus"):
        contents: Sequence[str] = corpus[start : start + batch_size]["contents"]
        encoded_batches.append(encoder.encode(contents, is_query=False))
    return np.asarray(
        np.concatenate(encoded_batches, axis=0),
        dtype=np.float32,
        order="C",
    )


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    faiss_type: str,
    faiss_gpu: bool,
):
    """Build an inner-product Flat or HNSW64 FAISS index."""

    if faiss_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(
            f"Unsupported faiss_type '{faiss_type}'. "
            f"Choose one of: {', '.join(SUPPORTED_INDEX_TYPES)}"
        )
    if faiss_gpu and faiss_type != "Flat":
        raise ValueError("--faiss_gpu is supported only with --faiss_type Flat")

    dimension = int(embeddings.shape[1])
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

    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)

    if faiss_gpu:
        index = faiss.index_gpu_to_cpu(index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an E5 Flat or HNSW64 FAISS index."
    )
    parser.add_argument(
        "--corpus_path",
        required=True,
        help="UTF-8 JSONL file; every row must contain id and contents.",
    )
    parser.add_argument(
        "--save_dir",
        default="index",
        help="Directory that will receive e5_<type>.index.",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--faiss_type",
        choices=SUPPORTED_INDEX_TYPES,
        default="Flat",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--corpus_num_proc", type=int, default=4)
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 for E5 encoding (CUDA only).",
    )
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build a Flat index on GPU (default). Use --no-faiss_gpu for "
            "CPU-only operation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing index with the same output name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.corpus_num_proc <= 0:
        raise ValueError("corpus_num_proc must be greater than zero")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"e5_{args.faiss_type}.index"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index already exists: {output_path}. Pass --overwrite to replace it."
        )

    print(f"Loading corpus: {Path(args.corpus_path).expanduser().resolve()}")
    corpus = load_corpus(args.corpus_path, num_proc=args.corpus_num_proc)
    print(f"Loaded {len(corpus):,} documents")

    print(f"Loading retrieval model: {args.model_path}")
    encoder = E5Encoder(
        args.model_path,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
    )
    print(f"Encoding device: {encoder.device}")
    embeddings = encode_corpus(corpus, encoder, batch_size=args.batch_size)

    faiss_device = "GPU" if args.faiss_gpu else "CPU"
    print(f"Building {args.faiss_type} inner-product index on {faiss_device}")
    index = build_faiss_index(
        embeddings,
        faiss_type=args.faiss_type,
        faiss_gpu=args.faiss_gpu,
    )
    if int(index.ntotal) != len(corpus):
        raise RuntimeError(
            f"Index contains {int(index.ntotal)} vectors; expected {len(corpus)}"
        )

    faiss.write_index(index, str(output_path))
    print(f"Saved {int(index.ntotal):,} vectors to: {output_path}")


if __name__ == "__main__":
    main()
