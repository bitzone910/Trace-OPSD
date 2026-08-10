"""Build the two-stage theorem search index bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from shared.config import DEFAULT_MODEL
from shared.corpus import load_corpus
from shared.encoder import E5Encoder
from shared.faiss_utils import SUPPORTED_INDEX_TYPES, create_faiss_index
from theorem_search_database.artifacts import (
    CONTENT_EMBEDDINGS_FILE,
    CONTENT_ROW_IDS_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    SLOGAN_INDEX_FILE,
    SLOGAN_OFFSETS_FILE,
)
from theorem_search_database.core import build_slogan_groups


def add_slogans_to_index(
    slogans: Sequence[str],
    encoder: E5Encoder,
    *,
    faiss_type: str,
    faiss_gpu: bool,
    batch_size: int,
):
    index = create_faiss_index(
        encoder.embedding_dimension,
        faiss_type=faiss_type,
        faiss_gpu=faiss_gpu,
    )
    for start in tqdm(
        range(0, len(slogans), batch_size),
        desc="Encoding unique slogans",
    ):
        batch = slogans[start : start + batch_size]
        embeddings = encoder.encode(batch, is_query=False)
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)
    return faiss.index_gpu_to_cpu(index) if faiss_gpu else index


def write_content_embeddings(
    corpus,
    encoder: E5Encoder,
    output_path: Path,
    *,
    batch_size: int,
) -> None:
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(corpus), encoder.embedding_dimension),
    )
    for start in tqdm(
        range(0, len(corpus), batch_size),
        desc="Encoding theorem contents",
    ):
        stop = min(start + batch_size, len(corpus))
        contents: Sequence[str] = corpus[start:stop]["contents"]
        matrix[start:stop] = encoder.encode(contents, is_query=False)
    matrix.flush()
    del matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a two-stage theorem search index bundle."
    )
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--save_dir", default="theorem_search_database/index")
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
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = save_dir / f"e5_{args.faiss_type}"
    if bundle_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index bundle already exists: {bundle_dir}. "
            "Pass --overwrite to replace it."
        )
    temporary_dir = save_dir / f".{bundle_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary_dir.mkdir()

    try:
        corpus = load_corpus(
            args.corpus_path,
            num_proc=args.corpus_num_proc,
            require_slogan=True,
        )
        print(f"Loaded {len(corpus):,} theorem records")
        groups = build_slogan_groups(corpus)
        unique_slogan_count = len(groups.unique_slogans)
        print(f"Grouped into {unique_slogan_count:,} unique slogans")

        encoder = E5Encoder(
            args.model_path,
            device=args.device,
            max_length=args.max_length,
            use_fp16=args.use_fp16,
        )
        slogan_index = add_slogans_to_index(
            groups.unique_slogans,
            encoder,
            faiss_type=args.faiss_type,
            faiss_gpu=args.faiss_gpu,
            batch_size=args.batch_size,
        )
        faiss.write_index(slogan_index, str(temporary_dir / SLOGAN_INDEX_FILE))
        del slogan_index
        np.save(temporary_dir / SLOGAN_OFFSETS_FILE, groups.offsets)
        np.save(temporary_dir / CONTENT_ROW_IDS_FILE, groups.content_row_ids)
        # The unique strings are no longer needed once the slogan index exists.
        # Releasing them before contents encoding matters for the 1.4M-row corpus.
        del groups
        write_content_embeddings(
            corpus,
            encoder,
            temporary_dir / CONTENT_EMBEDDINGS_FILE,
            batch_size=args.batch_size,
        )

        manifest = {
            "format_version": FORMAT_VERSION,
            "database_type": "theorem_search",
            "corpus_rows": len(corpus),
            "unique_slogans": unique_slogan_count,
            "embedding_dimension": encoder.embedding_dimension,
            "faiss_type": args.faiss_type,
            "model_path": args.model_path,
        }
        (temporary_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        os.replace(temporary_dir, bundle_dir)
        print(f"Saved theorem index bundle to: {bundle_dir}")
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


if __name__ == "__main__":
    main()
