"""Explicit-model, single-A800 greedy and multi-sample evaluation suite."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from trace_opsd.assets import (
    configure_asset_environment,
    resolve_assets_root,
    resolve_work_root,
)
from trace_opsd.compat import read_json

from .data import BY_NAME, load_snapshot, sha256_file
from .inventory import build_inventory
from .runner import EvalConfig
from .workflow import _write_json, report

SCHEMA = "trace_opsd.evaluation.suite.v2"
BENCHMARKS = tuple(BY_NAME)


class SuiteError(RuntimeError):
    """A recoverable input, launch, or result-integrity failure."""


def preflight(
    repo_root: Path, dataset_dir: Path, *, model_list: Path, assets_root: Path
) -> dict[str, Any]:
    """Collect all input errors without exporting weights or starting GPU services."""
    inventory = build_inventory(model_list, assets_root=assets_root)
    errors = list(inventory["errors"])
    datasets = {}
    try:
        snapshot, rows = load_snapshot(dataset_dir)
        datasets = {
            name: {
                "count": len(rows[name]),
                "sha256": snapshot["benchmarks"][name]["normalized_sha256"],
                "training_overlap_count": snapshot["benchmarks"][name].get(
                    "training_overlap_count", 0
                ),
            }
            for name in BENCHMARKS
        }
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"evaluation snapshot: {exc}")
    for relative in (
        "retrieval/general_search/index/e5_Flat.index",
        "retrieval/general_search/corpus/search_r1_corpus.jsonl",
        "models/e5-base-v2/config.json",
    ):
        path = assets_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty tool resource: {path}")
    retriever = assets_root / "models/e5-base-v2"
    if not any(retriever.glob("*.safetensors")) and not any(
        retriever.glob("pytorch_model*.bin")
    ):
        errors.append(f"retriever weights missing: {retriever}")
    for module in (
        "torch",
        "vllm",
        "transformers",
        "math_verify",
        "faiss",
        "httpx",
        "fastapi",
        "uvicorn",
        "psutil",
    ):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            errors.append(f"missing Python dependency in {sys.executable}: {module}")
    return {
        **inventory,
        "dataset_dir": str(dataset_dir),
        "datasets": datasets,
        "questions_per_model": sum(d["count"] for d in datasets.values()),
        "errors": errors,
        "ready": not errors,
    }


@contextlib.contextmanager
def suite_lock(output_root: Path) -> Iterator[None]:
    """OS locks survive stale files and are released even after power loss."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".suite.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SuiteError(
                    "another evaluation owns this output directory"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SuiteError(
                    "another evaluation owns this output directory"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_logged(command: list[str], log_file: Path, repo_root: Path) -> None:
    """Stream progress and reap only this command's owned process group."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
            start_new_session=os.name != "nt",
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            code = process.wait()
            if code:
                raise SuiteError(f"evaluation command exited {code}; see {log_file}")
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()


def validate_result(
    run_dir: Path, *, limit_per_benchmark: int | None = None
) -> dict[str, Any]:
    """A limited run is diagnostic; scoring errors cannot enter official reports."""
    try:
        manifest = read_json(run_dir / 'manifest.json')
        metrics = read_json(run_dir / 'metrics.json')
        if metrics["comparison_fingerprint"] != manifest["comparison_fingerprint"]:
            raise ValueError("manifest and metrics fingerprints differ")
        if metrics["predictions_sha256"] != sha256_file(run_dir / "predictions.jsonl"):
            raise ValueError("predictions checksum changed")
        if set(metrics["benchmarks"]) != set(BENCHMARKS):
            raise ValueError("result does not contain all five benchmarks")
        full = limit_per_benchmark is None or all(
            limit_per_benchmark >= spec.count for spec in BY_NAME.values()
        )
        if bool(metrics["complete_benchmarks"]) != full:
            raise ValueError("limited/full completeness flag is inconsistent")
        for name, stats in metrics["benchmarks"].items():
            count = min(limit_per_benchmark or BY_NAME[name].count, BY_NAME[name].count)
            if stats["total"] != count:
                raise ValueError(f"{name}: expected {count} rows, got {stats['total']}")
            if stats.get("verification_errors", 0):
                raise ValueError(f"{name}: mathematical verification errors")
            if limit_per_benchmark and stats.get("generation_errors", 0):
                raise ValueError(f"{name}: generation errors in limited diagnostic run")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SuiteError(f"invalid result at {run_dir}: {exc}") from exc
    return metrics


def run_phase(
    entries: list[dict[str, Any]],
    *,
    protocol: str,
    output_root: Path,
    repo_root: Path,
    dataset_dir: Path,
    gpu: str,
    assets_root: Path,
    work_root: Path,
    limit_per_benchmark: int | None,
    sample_config: EvalConfig,
    n_samples: int,
    pass_ks: tuple[int, ...],
) -> list[Path]:
    from .sampling_batch import validate_result as validate_sampled

    runs = []
    phase_root = output_root / ("limited" if limit_per_benchmark else "full") / protocol
    for index, entry in enumerate(entries, 1):
        run_dir = phase_root / entry["label"]
        logs_dir = output_root / "logs" / protocol / entry["label"]
        _write_json(
            output_root / "batch_status.json",
            {
                "schema": SCHEMA,
                "status": "running",
                "protocol": protocol,
                "model": entry["label"],
                "model_index": index,
                "models": len(entries),
            },
        )
        print(
            f"[{protocol} {index}/{len(entries)}] {entry['label']} on GPU {gpu}",
            flush=True,
        )
        command = [
            "bash",
            str(repo_root / "scripts/run_eval.sh"),
            "--model-path",
            entry["export_path"],
            "--run-dir",
            str(run_dir),
            "--model-gpus",
            gpu,
            "--retrieval-device",
            "cpu",
            "--assets-root",
            str(assets_root),
            "--work-root",
            str(work_root),
            "--dataset-dir",
            str(dataset_dir),
            "--benchmarks",
            ",".join(BENCHMARKS),
            "--mode",
            "tool",
            "--thinking-mode",
            "auto",
            "--concurrency",
            "16",
            "--logs-dir",
            str(logs_dir),
        ]
        if limit_per_benchmark:
            command.extend(["--limit-per-benchmark", str(limit_per_benchmark)])
        if protocol == "sampling8":
            command.extend(
                [
                    "--samples-per-question",
                    str(n_samples),
                    "--temperature",
                    str(sample_config.temperature),
                    "--top-p",
                    str(sample_config.top_p),
                    "--top-k",
                    str(sample_config.top_k),
                    "--seed",
                    str(sample_config.seed),
                    "--pass-ks",
                    ",".join(map(str, pass_ks)),
                ]
            )
        run_logged(command, logs_dir / "console.log", repo_root)
        if protocol == "sampling8":
            validate_sampled(
                run_dir,
                limit_per_benchmark=limit_per_benchmark,
                n_samples=n_samples,
                pass_ks=pass_ks,
            )
        else:
            validate_result(run_dir, limit_per_benchmark=limit_per_benchmark)
        runs.append(run_dir)
    return runs


def package_results(output_root: Path, *, complete: bool) -> Path:
    """Bundle result files only, never weights or a partially written tar."""
    archive_name = output_root.name + "-analysis.tar.gz"
    name = (
        archive_name
        if complete
        else archive_name.replace(".tar.gz", ".incomplete.tar.gz")
    )
    target = output_root.parent / name
    temporary = target.with_name(f".{name}.{uuid4().hex}.tmp")
    allowed = {".json", ".jsonl", ".csv", ".md", ".log", ".txt"}
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(output_root.rglob("*")):
                if path.suffix not in allowed or not path.is_file():
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(output_root):
                    raise SuiteError(f"refusing to archive an external file: {path}")
                archive.add(
                    path,
                    arcname=(
                        Path(output_root.name) / path.relative_to(output_root)
                    ).as_posix(),
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def run_suite(
    *,
    repo_root: Path,
    dataset_dir: Path,
    model_list: Path,
    output_root: Path,
    protocol: str,
    assets_root: Path | None = None,
    work_root: Path | None = None,
    gpu: str = "0",
    preflight_only: bool = False,
    limit_per_benchmark: int | None = None,
    n_samples: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    seed: int = 42,
    pass_ks: tuple[int, ...] = (1, 2, 4, 8),
) -> Path | None:
    from .sampling_batch import sampling_settings
    from .sampling_workflow import report_sampled_runs

    if not gpu.isdecimal():
        raise SuiteError("--gpu must name one numeric GPU ID")
    if protocol not in {"greedy", "sampling8", "all"}:
        raise SuiteError("explicitly select --protocol greedy, sampling8, or all")
    if limit_per_benchmark is not None and (
        type(limit_per_benchmark) is not int or limit_per_benchmark < 1
    ):
        raise SuiteError("--limit-per-benchmark must be a positive integer")
    sample_config, pass_ks = sampling_settings(
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        pass_ks=pass_ks,
    )
    assets_root, work_root = configure_asset_environment(assets_root, work_root)
    repo_root, dataset_dir, output_root = (
        repo_root.resolve(),
        dataset_dir.resolve(),
        output_root.resolve(),
    )
    if output_root.is_relative_to(repo_root):
        raise SuiteError("evaluation outputs must be outside the source repository")
    with suite_lock(output_root):
        try:
            inventory = preflight(
                repo_root, dataset_dir, model_list=model_list, assets_root=assets_root
            )
            for entry in inventory["models"]:
                print(
                    f"{entry['label']}: {entry['status']} — {entry['export_path']}",
                    flush=True,
                )
            if inventory["errors"]:
                raise SuiteError("preflight failed:\n" + "\n".join(inventory["errors"]))
            protocols = ("greedy", "sampling8") if protocol == "all" else (protocol,)
            suite = {
                "schema": SCHEMA,
                "dataset_dir": str(dataset_dir),
                "datasets": inventory["datasets"],
                "gpu": gpu,
                "assets_root": str(assets_root),
                "baseline": inventory["baseline"],
                "protocols": list(protocols),
                "limit_per_benchmark": limit_per_benchmark,
                "settings": {
                    name: {
                        "config": (
                            sample_config if name == "sampling8" else EvalConfig()
                        ).to_dict(),
                        "n_samples": n_samples if name == "sampling8" else 1,
                        "pass_ks": list(pass_ks) if name == "sampling8" else [1],
                    }
                    for name in protocols
                },
                "inference": {
                    "dtype": "bfloat16",
                    "tensor_parallel_size": 1,
                    "quantization": None,
                    "concurrency": 16,
                },
                "models": [
                    {
                        key: entry[key]
                        for key in (
                            "label",
                            "step",
                            "source_run",
                            "export_path",
                            "model_identity",
                        )
                    }
                    for entry in inventory["models"]
                ],
            }
            suite_path = output_root / "suite.json"
            if (
                suite_path.exists()
                and read_json(suite_path) != suite
            ):
                raise SuiteError("suite inputs changed; use a new --output-root")
            _write_json(output_root / "inventory.json", inventory)
            if preflight_only:
                print(
                    f"Preflight passed; inventory: {output_root / 'inventory.json'}",
                    flush=True,
                )
                return None
            _write_json(suite_path, suite)
            all_full = limit_per_benchmark is None or all(
                limit_per_benchmark >= spec.count for spec in BY_NAME.values()
            )
            for name in protocols:
                runs = run_phase(
                    inventory["models"],
                    protocol=name,
                    output_root=output_root,
                    repo_root=repo_root,
                    dataset_dir=dataset_dir,
                    gpu=gpu,
                    assets_root=assets_root,
                    work_root=work_root,
                    limit_per_benchmark=limit_per_benchmark,
                    sample_config=sample_config,
                    n_samples=n_samples,
                    pass_ks=pass_ks,
                )
                if all_full:
                    baseline = next(
                        (r for r in runs if r.name == inventory["baseline"]), None
                    )
                    summarize = report_sampled_runs if name == "sampling8" else report
                    summarize(
                        runs,
                        output_root / "comparison" / name,
                        baseline_run_dir=baseline,
                    )
            _write_json(
                output_root / "batch_status.json",
                {
                    "schema": SCHEMA,
                    "status": "complete" if all_full else "limited_complete",
                    "complete": all_full,
                    "models": len(inventory["models"]),
                    "protocols": list(protocols),
                    "questions_per_model": inventory["questions_per_model"],
                },
            )
            archive = package_results(output_root, complete=all_full)
            status = "complete" if all_full else "limited diagnostics complete"
            print(
                f"Evaluation {status}. Analysis archive: {archive}",
                flush=True,
            )
            return archive
        except (Exception, KeyboardInterrupt) as exc:
            _write_json(
                output_root / "batch_status.json",
                {
                    "schema": SCHEMA,
                    "status": "failed",
                    "complete": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if not preflight_only:
                try:
                    archive = package_results(output_root, complete=False)
                    print(
                        f"Incomplete diagnostics: {archive}",
                        file=sys.stderr,
                        flush=True,
                    )
                except (OSError, SuiteError):
                    pass
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("greedy", "sampling8", "all"), required=True
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/evaluation",
    )
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--name", help="Result directory name under work-root/evaluation"
    )
    output.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        help="Diagnostic subset; excluded from formal comparison",
    )
    parser.add_argument("--samples-per-question", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-ks", default="1,2,4,8")
    args = parser.parse_args(argv)
    if args.name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.name):
        parser.error("--name must be a safe result directory label")
    assets = resolve_assets_root(args.assets_root)
    work = resolve_work_root(args.work_root, assets_root=assets)
    target = args.output_root or work / "evaluation" / args.name
    if os.name != "nt":

        def interrupted(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        run_suite(
            repo_root=Path(__file__).resolve().parents[2],
            dataset_dir=args.dataset_dir,
            model_list=args.models_file,
            protocol=args.protocol,
            output_root=target,
            assets_root=assets,
            work_root=work,
            gpu=args.gpu,
            preflight_only=args.preflight_only,
            limit_per_benchmark=args.limit_per_benchmark,
            n_samples=args.samples_per_question,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            pass_ks=tuple(int(k) for k in args.pass_ks.split(",")),
        )
    except KeyboardInterrupt:
        print(
            "Evaluation interrupted; rerun the same command to resume.", file=sys.stderr
        )
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
