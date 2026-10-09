"""GPU and compiled-kernel validation used by ``scripts/setup_env.sh``."""

from __future__ import annotations

import argparse
import json
import os
import re
from importlib import import_module, metadata
from pathlib import Path

from trace_opsd.hardware_profile import HardwareProfile, load_hardware_profile


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _matches_profile(name: str, profile: HardwareProfile) -> bool:
    return any(
        re.search(pattern, name, re.IGNORECASE) for pattern in profile.name_patterns
    )


def _cache_paths(capability: tuple[int, int]) -> dict[str, str]:
    architecture = f"sm{capability[0]}{capability[1]}"
    variables = (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    )
    result: dict[str, str] = {}
    for variable in variables:
        configured = os.environ.get(variable)
        if not configured:
            raise RuntimeError(f"{variable} must be configured")
        path = Path(configured).resolve()
        if architecture not in path.parts:
            raise RuntimeError(f"{variable} is not scoped to {architecture}: {path}")
        path.mkdir(parents=True, exist_ok=True)
        result[variable] = str(path)
    return result


def _flash_attention_probe(torch: object) -> dict[str, object]:
    from flash_attn import flash_attn_func

    probes: dict[str, object] = {}
    for dtype_name, dtype in (
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ):
        query = torch.randn(
            (1, 32, 2, 128),
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)
        output = flash_attn_func(query, key, value, causal=True)
        output.float().square().mean().backward()
        if query.grad is None or key.grad is None or value.grad is None:
            raise RuntimeError(
                f"FlashAttention {dtype_name} backward produced no gradient"
            )
        probes[dtype_name] = {
            "shape": list(output.shape),
            "gradient_norm": float(query.grad.float().norm().item()),
        }
    return probes


def _xformers_probe(torch: object) -> dict[str, object]:
    from xformers.ops import memory_efficient_attention

    query = torch.randn(
        (1, 32, 2, 128),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    output = memory_efficient_attention(query, key, value)
    output.float().square().mean().backward()
    if query.grad is None:
        raise RuntimeError("xFormers backward produced no gradient")
    return {
        "shape": list(output.shape),
        "gradient_norm": float(query.grad.float().norm().item()),
    }


def _training_import_closure() -> list[str]:
    modules = (
        "trace_opsd.sft.cli",
        "llamafactory",
        "trace_opsd.grpo.agent_loop",
        "trace_opsd.grpo.trace_main",
        "trace_opsd.grpo.reward",
        "trace_opsd.retrieval.general.retrieval_server",
        "trace_opsd.tools.service",
        "verl.trainer.main_ppo",
        "verl.workers.rollout.vllm_rollout.vllm_rollout",
        "vllm._C",
        "flashinfer",
        "ray",
    )
    for module in modules:
        import_module(module)
    return list(modules)


def _imports(expected_gpu: str) -> None:
    profile = load_hardware_profile(expected_gpu)
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend != profile.vllm_attention_backend:
        raise RuntimeError(
            f"import closure requires {profile.vllm_attention_backend}, "
            f"found {backend!r}"
        )
    imported = _training_import_closure()
    print("SFT/TRACE import closure passed: " + ", ".join(imported))


def _verl_triton_probe(torch: object) -> dict[str, object]:
    from verl.utils.kernel.kernels import (
        efficient_entropy_backward,
        efficient_entropy_forward,
    )

    hidden = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1024, 128), device="cuda", dtype=torch.bfloat16)
    labels = torch.arange(8, device="cuda", dtype=torch.long)
    logprobs, entropy, maximum, accumulate, entropy_b = efficient_entropy_forward(
        hidden, weight, labels, reduction=2
    )
    d_hidden, d_weight = efficient_entropy_backward(
        torch.ones_like(logprobs),
        torch.ones_like(entropy),
        hidden,
        weight,
        labels,
        maximum,
        accumulate,
        entropy_b,
        reduction=2,
    )
    if not torch.isfinite(d_hidden).all() or not torch.isfinite(d_weight).all():
        raise RuntimeError("verl Triton entropy kernel returned non-finite gradients")
    return {
        "logprobs": float(logprobs.item()),
        "entropy_mean": float(entropy.mean().item()),
        "hidden_gradient_norm": float(d_hidden.float().norm().item()),
        "weight_gradient_norm": float(d_weight.float().norm().item()),
    }


def _torch_compile_probe(torch: object) -> dict[str, float]:
    inputs = torch.randn(
        (64, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        (128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def compiled_loss(left: object, right: object) -> object:
        return (left @ right).sin().float().square().mean()

    compiled = torch.compile(compiled_loss, fullgraph=True)
    loss = compiled(inputs, weight)
    loss.backward()
    if inputs.grad is None or weight.grad is None:
        raise RuntimeError("torch.compile backward produced no gradient")
    return {
        "loss": float(loss.item()),
        "input_gradient_norm": float(inputs.grad.float().norm().item()),
    }


def _single(expected_gpu: str, output: Path) -> None:
    import numpy as np
    import torch

    profile = load_hardware_profile(expected_gpu)
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend != profile.vllm_attention_backend:
        raise RuntimeError(
            "VLLM_ATTENTION_BACKEND must be "
            f"{profile.vllm_attention_backend}, found {backend!r}"
        )
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA 12.8 wheel, found {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("PyTorch cannot initialize a visible CUDA device")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if not _matches_profile(name, profile):
        raise RuntimeError(f"expected {expected_gpu}, found {name}")
    if capability != profile.compute_capability:
        raise RuntimeError(
            f"expected compute capability {profile.compute_capability}, "
            f"found {capability}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible GPU does not support BF16")

    cache_paths = _cache_paths(capability)
    compile_probe = _torch_compile_probe(torch)
    flash_probe = _flash_attention_probe(torch)
    xformers_probe = {"status": "passed", **_xformers_probe(torch)}

    import faiss

    resources = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(4)
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    vectors = np.eye(4, dtype="float32")
    gpu_index.add(vectors)
    distances, indices = gpu_index.search(vectors[:1], 1)
    if int(indices[0, 0]) != 0:
        raise RuntimeError("FAISS GPU search returned an unexpected nearest neighbor")

    import_module("verl")
    import_closure = _training_import_closure()
    verl_probe = _verl_triton_probe(torch)
    torch.cuda.synchronize()

    _atomic_json(
        output,
        {
            "status": "passed",
            "expected_gpu": expected_gpu,
            "gpu_name": name,
            "compute_capability": list(capability),
            "bf16_supported": True,
            "torch_cuda": torch.version.cuda,
            "cache_paths": cache_paths,
            "packages": {
                package: metadata.version(package)
                for package in (
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "triton",
                    "vllm",
                    "xformers",
                    "flash-attn",
                    "flashinfer-python",
                    "faiss-gpu-cu12",
                    "ray",
                    "verl",
                )
            },
            "faiss_result": {
                "index": int(indices[0, 0]),
                "score": float(distances[0, 0]),
            },
            "torch_compile": compile_probe,
            "flash_attention": flash_probe,
            "xformers": xformers_probe,
            "vllm_attention_backend": backend,
            "import_closure": import_closure,
            "verl_triton": verl_probe,
        },
    )


def _distributed(expected_gpu: str, output: Path) -> None:
    import gc

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    profile = load_hardware_profile(expected_gpu)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in profile.distributed_gpu_counts:
        raise RuntimeError(
            f"{expected_gpu} distributed validation requires one of "
            f"{profile.distributed_gpu_counts}, found {world_size} ranks"
        )

    torch.cuda.set_device(local_rank)
    name = torch.cuda.get_device_name(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    if not _matches_profile(name, profile) or capability != profile.compute_capability:
        raise RuntimeError(
            f"rank {rank} expected {expected_gpu} {profile.cache_arch}, "
            f"found {name} {capability}"
        )
    _cache_paths(capability)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        value = torch.tensor([rank + 1.0], device=local_rank)
        dist.all_reduce(value)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(
                f"NCCL all-reduce expected {expected_sum}, found {value.item()}"
            )

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(256, 16, bias=False),
        ).to(device=local_rank, dtype=torch.bfloat16)
        wrapped = FSDP(
            model,
            device_id=local_rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
        inputs = torch.randn((8, 128), device=local_rank, dtype=torch.bfloat16)
        loss = wrapped(inputs).float().square().mean()
        loss.backward()
        optimizer.step()
        dist.barrier(device_ids=[local_rank])
        loss_value = loss.item()
        if rank == 0:
            _atomic_json(
                output,
                {
                    "status": "passed",
                    "expected_gpu": expected_gpu,
                    "world_size": world_size,
                    "backend": dist.get_backend(),
                    "all_reduce_sum": value.item(),
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "loss": loss_value,
                },
            )
        del loss, inputs, optimizer, wrapped, model, value
        torch.cuda.synchronize(local_rank)
        gc.collect()
    finally:
        dist.destroy_process_group()


def _vllm_model(model: str, output: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend != "FLASH_ATTN":
        raise RuntimeError(f"vLLM model probe requires FLASH_ATTN, found {backend!r}")
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.60,
        trust_remote_code=False,
    )
    results = engine.generate(
        ["Return the number one:"],
        SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )
    token_count = len(results[0].outputs[0].token_ids)
    if token_count < 1:
        raise RuntimeError("vLLM returned an empty completion")
    _atomic_json(
        output,
        {
            "status": "passed",
            "model": model,
            "generated_tokens": token_count,
            "attention_backend": backend,
            "xformers_backend_forbidden": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("single", "distributed"):
        child = subparsers.add_parser(command)
        child.add_argument("--expected-gpu", choices=("a800",), required=True)
        child.add_argument("--output", type=Path, required=True)
    model_parser = subparsers.add_parser("vllm-model")
    model_parser.add_argument("--model", required=True)
    model_parser.add_argument("--output", type=Path, required=True)
    imports_parser = subparsers.add_parser("imports")
    imports_parser.add_argument("--expected-gpu", choices=("a800",), required=True)
    args = parser.parse_args()

    if args.command == "single":
        _single(args.expected_gpu, args.output)
    elif args.command == "distributed":
        _distributed(args.expected_gpu, args.output)
    elif args.command == "vllm-model":
        _vllm_model(args.model, args.output)
    else:
        _imports(args.expected_gpu)


if __name__ == "__main__":
    main()
