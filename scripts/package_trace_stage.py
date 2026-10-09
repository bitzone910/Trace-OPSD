"""Build an audited TRACE analysis bundle, optionally waiting for a running stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_analyzer():
    # scripts is a namespace directory: an installed regular package with the
    # same name can win even with the repository first on sys.path. Load the
    # exact sibling file instead of depending on that ambiguous package name.
    path = Path(__file__).resolve().with_name("analyze_trace_stage.py")
    if not path.is_file():
        raise ImportError(
            f"Missing TRACE analysis helper: {path}; "
            "sync scripts/analyze_trace_stage.py alongside package_trace_stage.py"
        )
    spec = importlib.util.spec_from_file_location(
        "_trace_opsd_trace_stage_analyzer", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TRACE analysis helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ANALYZER = _load_analyzer()
analyze = _ANALYZER.analyze
collect_training_metrics = _ANALYZER.collect_training_metrics
stage_work_root = _ANALYZER.stage_work_root
stage_checkpoint_root = _ANALYZER.stage_checkpoint_root
compare_validation = _ANALYZER.compare_validation
load_rows = _ANALYZER.load_rows
row_summary = _ANALYZER.row_summary
verify_receipts = _ANALYZER.verify_receipts
write_csv = _ANALYZER.write_csv
write_jsonl = _ANALYZER.write_jsonl
from trace_opsd.assets import configure_asset_environment  # noqa: E402
from trace_opsd.compat import metadata_path, read_json  # noqa: E402
from trace_opsd.grpo.credit_revision import (  # noqa: E402
    checkpoint_credit_state,
    resolve_trace_credit_config,
)
from trace_opsd.grpo.metrics_io import metric_coverage  # noqa: E402
from trace_opsd.grpo.turn_credit_io import read_turn_credit  # noqa: E402


def write_curves(path: Path, rows: list[dict]) -> None:
    """A self-contained interactive metric viewer; NaN/absent values leave gaps."""
    normalized = [
        {k: v if math.isfinite(v) else None for k, v in r.items()} for r in rows
    ]
    payload = json.dumps(normalized, allow_nan=False).replace("<", "\\u003c")
    template = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>TRACE stage metrics</title><style>
body{font:15px system-ui;margin:24px;background:#f6f8fc;color:#172238}
section{background:white;padding:18px;margin:14px 0;border-radius:10px;max-width:1000px}
canvas{width:100%;height:230px}select{max-width:100%;padding:8px}pre{white-space:pre-wrap}
</style><h1>TRACE 阶段指标</h1>
<p>每个点对应一个全局 step；缺失或非有限值显示为断点。
数据来源和完整性见 bundle_manifest.json。</p>
<section><label>全部指标：<select id="metric"></select></label>
<canvas id="custom" width="1000" height="250"></canvas>
<pre id="hover">移动鼠标查看 step 和数值。</pre></section><div id="overview"></div>
<script>const rows=__DATA__;
const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))].filter(k=>k!=='step').sort();
const select=document.getElementById('metric');keys.forEach(k=>{
const o=document.createElement('option');o.value=k;o.textContent=k;
select.appendChild(o)});
function draw(canvas,key){const ctx=canvas.getContext('2d');
const w=canvas.width,h=canvas.height;
ctx.clearRect(0,0,w,h);const points=rows.filter(r=>Number.isFinite(r[key]));
ctx.fillStyle='#172238';ctx.font='14px system-ui';ctx.fillText(key,80,22);
if(!points.length){ctx.fillText('没有可用数值',80,65);return}
let lo=Math.min(...points.map(r=>r[key])),hi=Math.max(...points.map(r=>r[key]));
if(hi===lo){const pad=Math.abs(lo)*.1||1;lo-=pad;hi+=pad}
const x0=Math.min(...rows.map(r=>r.step)),x1=Math.max(...rows.map(r=>r.step));
const X=s=>80+(s-x0)/Math.max(1,x1-x0)*(w-110),Y=v=>h-35-(v-lo)/(hi-lo)*(h-80);
ctx.strokeStyle='#d6deeb';ctx.fillStyle='#58677c';
for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=Y(v);ctx.beginPath();
ctx.moveTo(80,y);ctx.lineTo(w-30,y);ctx.stroke();
ctx.fillText(v.toPrecision(4),2,y+4)}
ctx.fillText('step '+x0,80,h-10);ctx.fillText('step '+x1,w-110,h-10);
ctx.strokeStyle='#2466cc';ctx.lineWidth=2;ctx.beginPath();let active=false;
rows.forEach(r=>{if(!Number.isFinite(r[key])){active=false;return}
if(active)ctx.lineTo(X(r.step),Y(r[key]));
else ctx.moveTo(X(r.step),Y(r[key]));active=true});ctx.stroke();
ctx.fillStyle='#2466cc';points.forEach(r=>{ctx.beginPath();
ctx.arc(X(r.step),Y(r[key]),2,0,Math.PI*2);ctx.fill()});
canvas.onmousemove=e=>{const ratio=(e.offsetX/canvas.clientWidth*w-80)/(w-110);
const s=x0+ratio*(x1-x0);
const r=rows.reduce((a,b)=>Math.abs(a.step-s)<Math.abs(b.step-s)?a:b);
document.getElementById('hover').textContent='step '+r.step+' | '+key+': '+
(r[key]??'缺失或非有限值')};}
select.onchange=()=>draw(document.getElementById('custom'),select.value);
select.value=keys.includes('trace/reference_kl/mean')?
'trace/reference_kl/mean':keys[0];if(keys.length)select.onchange();
['trace/reference_kl/mean','actor/kl_loss','actor/entropy','actor/grad_norm',
'actor/lr','response_length/mean','response_length/clip_ratio',
'trace/dynamic/candidate_acc_mean','trace/termination/invalid_action_limit_ratio',
'trace/token_credit_coverage'].forEach(k=>{
const s=document.createElement('section'),c=document.createElement('canvas');
c.width=1000;c.height=250;s.appendChild(c);
document.getElementById('overview').appendChild(s);draw(c,k)});
</script></html>"""
    path.write_text(template.replace("__DATA__", payload), encoding="utf-8")


def stage_ready(run_dir: Path, generation_dir: Path, step: int) -> bool:
    return all(
        p.is_file()
        for p in (
            run_dir / f"global_step_{step}/manifest.json",
            run_dir / f"trace_validation_step_{step}.json",
            generation_dir / f"validation/{step}.jsonl",
        )
    )


def make_bundle(
    *,
    repo: Path,
    data_root: Path | None = None,
    work_root: Path | None = None,
    run_name: str,
    after_step: int,
    step: int,
    log_files: list[Path],
    run_dir: Path | None = None,
    archive_dir: Path | None = None,
) -> tuple[Path, dict]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("invalid run name")
    if not 0 <= after_step < step:
        raise ValueError("require 0 <= after-step < step")
    repo = repo.resolve()
    work = stage_work_root(data_root, work_root)
    run_dir = (
        run_dir or stage_checkpoint_root(repo, data_root, work) / run_name
    ).resolve()
    generation = work / "grpo" / run_name
    if not stage_ready(run_dir, generation, step):
        raise ValueError(f"step {step} checkpoint/validation is not finished")
    archive_dir = (
        archive_dir or (repo if data_root is not None else work / "archives")
    ).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    analysis_root = (repo if data_root is not None else work) / "analysis" / run_name
    analysis_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".step-{step}-", dir=analysis_root
    ) as temporary:
        stage = Path(temporary) / f"step-{step}"
        summary = analyze(
            repo=repo,
            data_root=data_root,
            work_root=work_root,
            run_name=run_name,
            after_step=after_step,
            step=step,
            log_files=log_files,
            output=stage,
            metrics_files=[run_dir / "metrics.jsonl"],
            run_dir=run_dir,
        )
        with (stage / "training_metrics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            metrics = [
                {k: float(v) for k, v in row.items() if v != ""}
                for row in csv.DictReader(handle)
            ]
        records = {
            int(r["step"]): {k: v for k, v in r.items() if k != "step"} for r in metrics
        }
        normalized = [
            {
                "step": int(r["step"]),
                "data": {
                    k: v
                    if math.isfinite(v)
                    else str(v).replace("nan", "NaN").replace("inf", "Infinity")
                    for k, v in r.items()
                    if k != "step"
                },
            }
            for r in metrics
        ]
        write_jsonl(stage / "training_metrics.jsonl", normalized)
        write_curves(
            stage / "training_curves.html",
            [
                {"step": number, **records.get(number, {})}
                for number in range(after_step + 1, step + 1)
            ],
        )
        baseline = load_rows(generation / "validation/0.jsonl", 0)
        current = load_rows(generation / f"validation/{step}.jsonl", step)
        if after_step:
            previous = load_rows(
                generation / f"validation/{after_step}.jsonl", after_step
            )
            verify_receipts(
                run_dir,
                after_step,
                {"baseline": row_summary(baseline), "current": row_summary(previous)},
            )
            delta, pairs, changed = compare_validation(previous, current)
            write_csv(
                stage / "stage_validation_comparison.csv",
                pairs,
                ("question_key", "transition"),
            )
            write_jsonl(stage / "stage_changed_validation_questions.jsonl", changed)
            (stage / "stage_validation_summary.json").write_text(
                json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with (stage / "report.md").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n## 本阶段增量：step {after_step} → {step}\n\n"
                    f"正确率变化 {delta['delta_percentage_points']:+.4f} 个百分点；"
                    f"逐题变化 {delta['transitions']}。\n"
                )
        # Include exact validation outputs and accepted training trajectories,
        # but never model/optimizer weights or unrelated runs.
        for number in sorted({0, after_step, step}):
            destination = stage / f"raw/validation/{number}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generation / f"validation/{number}.jsonl", destination)
        for number in range(after_step + 1, step + 1):
            source = generation / f"rollouts/{number}.jsonl"
            if source.is_file():
                destination = stage / f"raw/rollouts/{number}.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        names = [
            "trace_opsd_grpo_run.json",
            "trace_curriculum.json",
            "initial_model_validation.json",
            "validation_verifier_revision.json",
            f"trace_validation_step_{step}.json",
            "best_checkpoint.json",
            "latest_checkpointed_iteration.txt",
        ]
        if after_step:
            names.append(f"trace_validation_step_{after_step}.json")
        for name in names:
            source = metadata_path(run_dir / name)
            if source.is_file():
                destination = stage / "run_metadata" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        manifest_path = run_dir / f"global_step_{step}/manifest.json"
        shutil.copyfile(manifest_path, stage / "checkpoint_manifest.json")
        for index, source in enumerate(log_files):
            if source.is_file():
                destination = stage / "logs" / f"{index}-{source.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        with (stage / "training_rollouts.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rollout_counts = {
                int(row["step"]): int(row["rows"]) for row in csv.DictReader(handle)
            }
        mismatches = [
            s
            for s, count in rollout_counts.items()
            if count != records.get(s, {}).get("trace/dynamic/accepted_rollouts", 256)
        ]
        coverage = metric_coverage(records, after_step, step)
        diagnostic_missing, diagnostic_invalid = [], []
        enabled_steps = set(coverage["python_error_penalty"]["enabled_steps"])
        metadata = metadata_path(run_dir / "trace_opsd_grpo_run.json")
        if metadata.is_file() and "trace_opsd" in read_json(metadata):
            for number in range(after_step + 1, step + 1):
                if resolve_trace_credit_config(run_dir, number)[
                    "python_error_penalty_enabled"
                ]:
                    enabled_steps.add(number)
                    if (
                        records.get(number, {}).get(
                            "trace/python_error_penalty/enabled"
                        )
                        != 1
                    ):
                        diagnostic_invalid.append(number)
        for number in sorted(enabled_steps):
            source = run_dir / "trace_turn_credit" / f"{number}.json"
            if not source.is_file():
                diagnostic_missing.append(number)
                continue
            try:
                diagnostic = read_turn_credit(source, step=number)
                rows = load_rows(generation / f"rollouts/{number}.jsonl", number)
                indexed = {r.get("trace_trajectory_index"): r for r in rows}
                if None in indexed or len(indexed) != len(rows):
                    raise ValueError("rollouts lack unique trace_trajectory_index")
                observed = records[number]["trace/python_error_penalty/observed_turns"]
                if len(diagnostic["turns"]) != observed:
                    raise ValueError("turn credit count differs from scalar metrics")
                if metadata.is_file() and diagnostic[
                    "credit_state"
                ] != checkpoint_credit_state(run_dir, number):
                    raise ValueError("turn credit revision differs from run history")
                seen = set()
                for turn in diagnostic["turns"]:
                    if any(
                        isinstance(turn.get(key), bool)
                        or not isinstance(turn.get(key), int)
                        or turn[key] < 0
                        for key in (
                            "trajectory_index",
                            "turn_index",
                            "response_start",
                            "response_end",
                        )
                    ):
                        raise ValueError(
                            "turn credit indices must be nonnegative integers"
                        )
                    key = (turn["trajectory_index"], turn["turn_index"])
                    row = indexed[turn["trajectory_index"]]
                    if key in seen or turn["uid"] != row.get("trajectory_uid"):
                        raise ValueError("turn credit is not aligned with rollout")
                    seen.add(key)
                    if not 0 <= turn["response_start"] < turn["response_end"]:
                        raise ValueError("invalid turn credit span")
                    if turn["response_end"] > row["trace_response_token_count"]:
                        raise ValueError("turn credit span exceeds rollout length")
                    original = row["trace_tool_turns"][turn["turn_index"]]
                    if original.get("tool_type") != "python" or any(
                        original.get(key) != turn.get(key)
                        for key in (
                            "response_start",
                            "response_end",
                            "status",
                            "error_category",
                            "error_origin",
                            "execution_result_version",
                            "transport_error",
                        )
                    ):
                        raise ValueError("turn credit differs from original tool turn")
                    if not any(
                        start <= turn["response_start"] < turn["response_end"] <= end
                        for start, end in row["trace_assistant_spans"]
                    ):
                        raise ValueError("turn credit includes non-assistant tokens")
                    if any(
                        not math.isfinite(v)
                        for k, v in turn.items()
                        if k.endswith(("_mean", "_min", "_max"))
                    ):
                        raise ValueError("nonfinite turn credit summary")
                expected_python = {
                    (index, turn_index)
                    for index, row in indexed.items()
                    for turn_index, original in enumerate(row["trace_tool_turns"])
                    if original.get("tool_type") == "python"
                }
                if seen != expected_python:
                    raise ValueError("rollout Python turns lack credit diagnostics")
                destination = stage / "raw/trace_turn_credit" / f"{number}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                diagnostic_invalid.append(number)
        complete = (
            coverage["complete"]
            and not summary["missing_rollout_steps"]
            and not mismatches
            and not diagnostic_missing
            and not diagnostic_invalid
            and not any(coverage["nonfinite_by_metric"].values())
        )
        audit = {
            "schema_version": "trace_opsd.trace.analysis-bundle.v1",
            "run_name": run_name,
            "after_step": after_step,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": complete,
            "metric_coverage": coverage,
            "missing_rollout_steps": summary["missing_rollout_steps"],
            "rollout_count_mismatch_steps": mismatches,
            "python_error_credit": {
                "availability": "available"
                if enabled_steps
                else coverage["python_error_penalty"]["availability"],
                "enabled_steps": sorted(enabled_steps),
                "missing_diagnostic_steps": diagnostic_missing,
                "invalid_diagnostic_steps": sorted(set(diagnostic_invalid)),
            },
            "sources": summary["sources"],
            "warnings": summary["warnings"],
            "files": [],
        }
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                audit["files"].append(
                    {
                        "path": path.relative_to(stage).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (stage / "bundle_manifest.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        suffix = "" if complete else ".incomplete"
        archive = archive_dir / f"{run_name}-step{step}-analysis{suffix}.tar.gz"
        fd, temp_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive_dir)
        os.close(fd)
        try:
            with tarfile.open(temp_name, "w:gz") as handle:
                handle.add(stage, arcname=f"step-{step}")
            # Read back and check every recorded member before publishing.
            with tarfile.open(temp_name, "r:gz") as handle:
                for item in audit["files"]:
                    content = handle.extractfile(f"step-{step}/{item['path']}").read()
                    if hashlib.sha256(content).hexdigest() != item["sha256"]:
                        raise ValueError(f"archive verification failed: {item['path']}")
            os.replace(temp_name, archive)
        finally:
            if Path(temp_name).exists():
                Path(temp_name).unlink()
    return archive, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--after-step", required=True, type=int)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--data-root", type=Path, help="historical DATA/runs layout")
    roots.add_argument("--work-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--log-file", type=Path, action="append")
    parser.add_argument(
        "--wait", action="store_true", help="wait for a stage already running"
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=43200)
    parser.add_argument("--metrics-grace-seconds", type=float, default=180)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("invalid run name")
    if (
        not 0 <= args.after_step < args.step
        or args.poll_seconds <= 0
        or args.timeout_seconds <= 0
        or args.metrics_grace_seconds < 0
        or not all(
            math.isfinite(v)
            for v in (
                args.poll_seconds,
                args.timeout_seconds,
                args.metrics_grace_seconds,
            )
        )
    ):
        parser.error("invalid step range or poll interval")
    configure_asset_environment(args.assets_root, args.work_root)
    work = stage_work_root(args.data_root, args.work_root)
    run_dir = (
        args.run_dir
        or stage_checkpoint_root(args.project_root, args.data_root, work)
        / args.run_name
    )
    logs = args.log_file or [
        work / "logs" / f"{args.run_name}-stage-{args.after_step}.log",
        run_dir / f"console-stage-{args.after_step}.log",
    ]
    generation = work / "grpo" / args.run_name
    if args.wait:
        print(
            f"Waiting for {args.run_name}, steps {args.after_step + 1}-{args.step}",
            flush=True,
        )
        started, receipt_seen = time.monotonic(), None
        while True:
            if stage_ready(run_dir, generation, args.step):
                if receipt_seen is None:
                    receipt_seen = time.monotonic()
                    print(
                        "Checkpoint and validation saved; checking metric coverage...",
                        flush=True,
                    )
                records, _, _ = collect_training_metrics(
                    repo=args.project_root,
                    data_root=args.data_root,
                    work_root=args.work_root,
                    run_name=args.run_name,
                    after_step=args.after_step,
                    step=args.step,
                    log_files=logs,
                    metrics_files=[run_dir / "metrics.jsonl"],
                )
                if metric_coverage(records, args.after_step, args.step)["complete"]:
                    # Receipt precedes background JSONL completion; do not copy
                    # a partially written trajectory/validation file.
                    try:
                        for number in range(args.after_step + 1, args.step + 1):
                            rows = load_rows(
                                generation / f"rollouts/{number}.jsonl", number
                            )
                            expected = records.get(number, {}).get(
                                "trace/dynamic/accepted_rollouts", 256
                            )
                            if len(rows) != expected:
                                raise ValueError("rollout JSONL not yet complete")
                        validation_rows = load_rows(
                            generation / f"validation/{args.step}.jsonl", args.step
                        )
                        receipt = json.loads(
                            (
                                run_dir / f"trace_validation_step_{args.step}.json"
                            ).read_text(encoding="utf-8")
                        )
                        if len(validation_rows) != receipt["validation"]["samples"]:
                            raise ValueError("validation JSONL not yet complete")
                        break
                    except (OSError, ValueError):
                        pass
                if time.monotonic() - receipt_seen >= args.metrics_grace_seconds:
                    print(
                        "Metric grace expired; missing data labelled incomplete.",
                        flush=True,
                    )
                    break
            if time.monotonic() - started >= args.timeout_seconds:
                parser.exit(
                    1, "Timed out waiting for completed checkpoint/validation.\n"
                )
            time.sleep(args.poll_seconds)
    try:
        archive, audit = make_bundle(
            repo=args.project_root,
            data_root=args.data_root,
            work_root=args.work_root,
            run_name=args.run_name,
            after_step=args.after_step,
            step=args.step,
            log_files=logs,
            run_dir=run_dir,
            archive_dir=args.archive_dir,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Analysis packaging failed; checkpoint unchanged: {exc}\n")
    print(f"Analysis archive: {archive}", flush=True)
    print(
        f"Complete metrics: {audit['complete']}; coverage: {audit['metric_coverage']}",
        flush=True,
    )
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
