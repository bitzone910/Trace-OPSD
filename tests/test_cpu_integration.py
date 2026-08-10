"""Opt-in CPU smoke test using the real local E5 model and FAISS runtime."""

from __future__ import annotations

import json
import os

import pytest


@pytest.mark.integration
def test_general_and_theorem_cpu_smoke(tmp_path) -> None:
    if os.environ.get("RUN_RETRIEVAL_INTEGRATION") != "1":
        pytest.skip("set RUN_RETRIEVAL_INTEGRATION=1 to run the CPU smoke test")

    faiss = pytest.importorskip("faiss")
    pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")

    import numpy as np

    from general_search_database.build_index import encode_corpus
    from general_search_database.retriever import DenseRetriever
    from shared.config import DEFAULT_MODEL, PROJECT_ROOT
    from shared.corpus import load_corpus
    from shared.encoder import E5Encoder
    from shared.faiss_utils import build_faiss_index
    from theorem_search_database.artifacts import (
        CONTENT_EMBEDDINGS_FILE,
        CONTENT_ROW_IDS_FILE,
        FORMAT_VERSION,
        MANIFEST_FILE,
        SLOGAN_INDEX_FILE,
        SLOGAN_OFFSETS_FILE,
    )
    from theorem_search_database.build_index import (
        add_slogans_to_index,
        write_content_embeddings,
    )
    from theorem_search_database.core import build_slogan_groups
    from theorem_search_database.retriever import TheoremRetriever

    encoder = E5Encoder(DEFAULT_MODEL, device="cpu", use_fp16=False)

    general_corpus_path = (
        PROJECT_ROOT / "general_search_database" / "corpus" / "sample_corpus.jsonl"
    )
    general_corpus = load_corpus(general_corpus_path, num_proc=1)
    general_embeddings = encode_corpus(general_corpus, encoder, batch_size=2)
    general_index = build_faiss_index(
        general_embeddings,
        faiss_type="Flat",
        faiss_gpu=False,
    )
    general_index_path = tmp_path / "general.index"
    faiss.write_index(general_index, str(general_index_path))
    general = DenseRetriever(
        general_index_path,
        general_corpus_path,
        model_path=DEFAULT_MODEL,
        device="cpu",
        faiss_gpu=False,
        corpus_num_proc=1,
    )
    assert len(general.search("mathematics", topk=99)) == len(general_corpus)

    theorem_corpus_path = (
        PROJECT_ROOT / "theorem_search_database" / "corpus" / "sample_corpus.jsonl"
    )
    theorem_corpus = load_corpus(
        theorem_corpus_path,
        num_proc=1,
        require_slogan=True,
    )
    groups = build_slogan_groups(theorem_corpus)
    theorem_bundle = tmp_path / "theorem_bundle"
    theorem_bundle.mkdir()
    slogan_index = add_slogans_to_index(
        groups.unique_slogans,
        encoder,
        faiss_type="Flat",
        faiss_gpu=False,
        batch_size=2,
    )
    faiss.write_index(slogan_index, str(theorem_bundle / SLOGAN_INDEX_FILE))
    np.save(theorem_bundle / SLOGAN_OFFSETS_FILE, groups.offsets)
    np.save(theorem_bundle / CONTENT_ROW_IDS_FILE, groups.content_row_ids)
    write_content_embeddings(
        theorem_corpus,
        encoder,
        theorem_bundle / CONTENT_EMBEDDINGS_FILE,
        batch_size=2,
    )
    (theorem_bundle / MANIFEST_FILE).write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "database_type": "theorem_search",
                "corpus_rows": len(theorem_corpus),
                "unique_slogans": len(groups.unique_slogans),
                "embedding_dimension": encoder.embedding_dimension,
                "faiss_type": "Flat",
                "model_path": DEFAULT_MODEL,
            }
        ),
        encoding="utf-8",
    )
    theorem = TheoremRetriever(
        theorem_bundle,
        theorem_corpus_path,
        model_path=DEFAULT_MODEL,
        device="cpu",
        faiss_gpu=False,
        corpus_num_proc=1,
        slogan_topk=20,
    )
    assert len(theorem.search("flat localization", topk=99)) == len(
        theorem_corpus
    )
