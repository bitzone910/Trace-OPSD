"""Build the general query-to-contents E5 FAISS index."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from general_search_database.retriever import load_corpus
from shared.config import DEFAULT_MODEL
from shared.encoder import E5Encoder
from shared.faiss_utils import SUPPORTED_INDEX_TYPES, build_faiss_index


def encode_corpus(
    corpus,
    encoder: E5Encoder,
    *,
    batch_size: int,
) -> np.ndarray:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the general E5 Flat or HNSW64 FAISS index."
    )
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--save_dir", default="general_search_database/index")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--faiss_type",
        choices=SUPPORTED_INDEX_TYPES,
        default="Flat",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--corpus_num_proc", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--faiss_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"e5_{args.faiss_type}.index"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index already exists: {output_path}. Pass --overwrite to replace it."
        )

    corpus = load_corpus(args.corpus_path, num_proc=args.corpus_num_proc)
    print(f"Loaded {len(corpus):,} documents")
    encoder = E5Encoder(
        args.model_path,
        device=args.device,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
    )
    embeddings = encode_corpus(corpus, encoder, batch_size=args.batch_size)
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
