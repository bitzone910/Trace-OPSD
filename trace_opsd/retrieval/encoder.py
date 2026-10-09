"""Search-R1-compatible E5 text encoding."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_MODEL


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
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool non-padding tokens, matching Search-R1's E5 behavior."""

    masked_hidden = last_hidden_state.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    token_counts = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return masked_hidden.sum(dim=1) / token_counts


class E5Encoder:
    """Hugging Face encoder with E5 query/passage prefixes."""

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
        self._encode_lock = RLock()
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
            raise ValueError(f"Model does not expose config.hidden_size: {model_path}")
        self.embedding_dimension = int(hidden_size)

    @torch.inference_mode()
    def encode(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool,
    ) -> np.ndarray:
        with self._encode_lock:
            return self._encode_unlocked(texts, is_query=is_query)

    def _encode_unlocked(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool,
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
        inputs = self.tokenizer(
            [f"{prefix}{text}" for text in text_list],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**device_inputs, return_dict=True)
        embeddings = mean_pool(
            output.last_hidden_state,
            device_inputs["attention_mask"],
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        array = np.asarray(
            embeddings.detach().float().cpu().numpy(),
            dtype=np.float32,
            order="C",
        )

        return array
