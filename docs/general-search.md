# General Search

General Search uses a Search-R1-format corpus with `id` and `contents` on each line.
E5 encodes queries and contents; FAISS performs inner-product retrieval. Retrieval
defaults to the CPU and serves TRACE and evaluation. Keep the corpus row order,
E5 model, and FAISS index versions consistent.

The complete index is published in `BitZone910/wiki_math`: 573,011 vectors of
dimension 768, stored as IndexFlatIP. Its SHA256 is
`09bf053b50922ad283620b4fdf84b4ab37e824a7cfd7cfb61e310c8452d7afb0`.
Download the pinned index and matching corpus from the repository root:

```bash
hf download BitZone910/wiki_math index/e5_Flat.index index/index-metadata.json \
  --repo-type dataset --revision 157dd03b9245158741b066ee00832759261e663c \
  --local-dir ../assets/retrieval/general_search
hf download BitZone910/wiki_math search_r1_corpus.jsonl \
  --repo-type dataset --revision 157dd03b9245158741b066ee00832759261e663c \
  --local-dir ../assets/retrieval/general_search/corpus
```

The encoder is pinned to `intfloat/e5-base-v2` revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Check file hashes and corpus ordering
before retrieval. Metadata resides in the downloaded `index/index-metadata.json`.
The published file is the existing index; embeddings were not regenerated.
The historical build's `max_length` and device were not independently recorded.

Rebuild only explicitly, without overwriting an existing index:

```bash
python -m trace_opsd.retrieval.general.build_index \
  --corpus_path ../assets/retrieval/general_search/corpus/search_r1_corpus.jsonl \
  --save_dir ../assets/retrieval/general_search/index \
  --model_path ../assets/models/e5-base-v2 \
  --faiss_type Flat --device cpu --no-faiss_gpu
```

The complete corpus and model are external assets. A small interface-check sample
remains in `data/general_search/corpus/sample_corpus.jsonl`. The index was verified
against local content before publication. On 2026-10-09, the local index, corpus,
and E5 copy were removed as requested. Restore these pinned inputs before running
retrieval locally.
