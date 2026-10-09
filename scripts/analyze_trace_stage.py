"""Export saved TRACE metrics and compare paired validation results (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_opsd.assets import (  # noqa: E402
    configure_asset_environment,
    resolve_work_root,
)
from trace_opsd.grpo.metrics_io import (  # noqa: E402
    merge_metrics,
    metric_coverage,
    read_metrics,
    recover_wandb,
)

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"\bstep:\s*(\d+)\s+-\s+")
METRIC = re.compile(
    r"(?:^|\s+-\s+)([A-Za-z_][\w/@.+-]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))",
    re.IGNORECASE,
)
WATCH = (
    "trace/dynamic/candidate_acc_mean",
    "trace/dynamic/accepted_acc_mean",
    "trace/dynamic/acceptance_ratio",
    "trace/dynamic/candidate_batches",
    "response_length/mean",
    "response_length/clip_ratio",
    "actor/entropy",
    "trace/termination/invalid_action_limit_ratio",
    "response/aborted_ratio",
    "trace/reference_kl/mean",
    "actor/grad_norm",
    "actor/lr",
    "actor/optimizer_update_success",
    "trace/token_credit_coverage",
    "trace/skipped_missing_solution_hindsight",
    "trace/skipped_view_too_long",
    "trace/adv/abs_delta_mean",
    "timing_s/step",
    "trace/python_error_penalty/token_ratio",
    "trace/python_error_penalty/penalty_abs_mean",
    "trace/python_error_penalty/positive_to_negative_ratio",
    "trace/python_error_penalty/code_success_turns",
    "trace/python_error_penalty/code_success_ratio",
    "tool_call_counts/mean",
    "trace/python_error_penalty/error_turns",
)
DIAGNOSTICS = (
    "answer_present",
    "tool_call_counts",
    "python_call_count",
    "tool_error_count",
    "invalid_action_count",
)


def parse_metrics(
    text: str, *, rejected: list[str] | None = None
) -> tuple[dict[int, dict[str, float]], list[int]]:
    """Parse console records; recovery callers may reject ambiguous chunks."""
    text = ANSI.sub("", text)
    starts = list(STEP.finditer(text))
    records: dict[int, dict[str, float]] = {}
    conflicts: set[int] = set()
    for index, start in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        pairs = METRIC.findall(text[start.end() : stop])
        values = {key: float(value) for key, value in pairs}
        step = int(start.group(1))
        # Console output can be interrupted/interleaved. Never reassign a
        # mismatched chunk to its inner step or mix its values into good data.
        # Check every occurrence, not just the final duplicate key.
        if any(
            key == "training/global_step" and float(value) != step
            for key, value in pairs
        ):
            message = f"console step {step} disagrees with training/global_step"
            if rejected is None:
                raise ValueError(message)
            rejected.append(message + "; ambiguous record ignored")
            continue
        previous = records.setdefault(step, {})
        if any(
            key in previous and previous[key] != value for key, value in values.items()
        ):
            conflicts.add(step)
        previous.update(values)
    return records, sorted(conflicts)


def load_rows(path: Path, step: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not {"input", "output", "gts", "score", "step"} <= row.keys()
            ):
                raise ValueError(f"{path}:{line_number}: missing generation fields")
            if (
                row["step"] != step
                or isinstance(row["score"], bool)
                or row["score"] not in (0, 1)
            ):
                raise ValueError(f"{path}:{line_number}: invalid step or binary score")
            if not isinstance(row["input"], str) or not isinstance(row["output"], str):
                raise ValueError(f"{path}:{line_number}: input/output must be strings")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty generation file: {path}")
    return rows


def numeric_mean(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows]
    if not values or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
    ):
        return None
    return mean(values)


def row_summary(rows: list[dict]) -> dict:
    lengths = sorted(len(row["output"]) for row in rows)
    return {
        "rows": len(rows),
        "correct": sum(row["score"] for row in rows),
        "pass_at_1": numeric_mean(rows, "score"),
        **{key: numeric_mean(rows, key) for key in DIAGNOSTICS},
        "output_chars_mean": mean(lengths),
        "output_chars_p90": lengths[math.ceil(0.9 * len(lengths)) - 1],
        "output_chars_correct_mean": numeric_mean(
            [{"length": len(r["output"])} for r in rows if r["score"] == 1], "length"
        ),
        "output_chars_incorrect_mean": numeric_mean(
            [{"length": len(r["output"])} for r in rows if r["score"] == 0], "length"
        ),
        "termination_counts": dict(
            Counter(str(r.get("termination_reason", "unknown")) for r in rows)
        ),
    }


def question_key(row: dict) -> str:
    content = json.dumps([row["input"], row["gts"]], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compare_validation(
    baseline: list[dict], current: list[dict]
) -> tuple[dict, list[dict], list[dict]]:
    base_by_key = {question_key(row): row for row in baseline}
    curr_by_key = {question_key(row): row for row in current}
    if len(base_by_key) != len(baseline) or len(curr_by_key) != len(current):
        raise ValueError(
            "duplicate validation input+ground truth: pairing is ambiguous"
        )
    if base_by_key.keys() != curr_by_key.keys():
        raise ValueError(
            "baseline and current validation contain different questions/ground truths"
        )
    pairs, changed = [], []
    counts: Counter = Counter()
    for key, before in base_by_key.items():
        after = curr_by_key[key]
        status = {
            (0, 0): "both_wrong",
            (0, 1): "wrong_to_right",
            (1, 0): "right_to_wrong",
            (1, 1): "both_right",
        }[before["score"], after["score"]]
        counts[status] += 1
        pair = {
            "question_key": key,
            "transition": status,
            "baseline_score": before["score"],
            "current_score": after["score"],
        }
        for prefix, row in (("baseline", before), ("current", after)):
            pair.update({f"{prefix}_{name}": row.get(name) for name in DIAGNOSTICS})
            pair[f"{prefix}_output_chars"] = len(row["output"])
        pairs.append(pair)
        if before["score"] != after["score"]:
            changed.append(
                {
                    **pair,
                    "input": before["input"],
                    "gts": before["gts"],
                    "baseline_output": before["output"],
                    "current_output": after["output"],
                }
            )
    gains, losses = counts["wrong_to_right"], counts["right_to_wrong"]
    discordant = gains + losses
    # Exact two-sided McNemar test, conditional on discordant paired binary outcomes.
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    summary = {
        "baseline": row_summary(baseline),
        "current": row_summary(current),
        "delta_percentage_points": 100 * (gains - losses) / len(baseline),
        "transitions": {
            key: counts[key]
            for key in ("both_right", "both_wrong", "wrong_to_right", "right_to_wrong")
        },
        "paired_mcnemar_exact_p": p_value,
    }
    return summary, pairs, changed


def verify_receipts(run_dir: Path, step: int, comparison: dict) -> None:
    before = json.loads(
        (run_dir / "initial_model_validation.json").read_text(encoding="utf-8")
    )
    after = json.loads(
        (run_dir / f"trace_validation_step_{step}.json").read_text(encoding="utf-8")
    )
    if before.get("step") != 0 or after.get("step") != step:
        raise ValueError("validation receipt step mismatch")
    if not isinstance(before.get("validation"), dict) or before[
        "validation"
    ] != after.get("validation"):
        raise ValueError("baseline/current receipt validation descriptors differ")
    expected_rows = before["validation"].get("samples")
    if expected_rows is not None and any(
        comparison[name]["rows"] != expected_rows for name in ("baseline", "current")
    ):
        raise ValueError("validation receipt sample count disagrees with JSONL")
    for receipt, name in ((before, "baseline"), (after, "current")):
        score = receipt.get("pass_at_1")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not math.isclose(
                score, comparison[name]["pass_at_1"], abs_tol=1e-12, rel_tol=0
            )
        ):
            raise ValueError(f"{name} receipt pass@1 disagrees with JSONL results")


def write_csv(path: Path, rows: list[dict], first: tuple[str, ...] = ()) -> None:
    fields = list(first) + sorted({key for row in rows for key in row} - set(first))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def stage_work_root(data_root: Path | None, work_root: Path | None) -> Path:
    """Explicit old data_root keeps its historical /runs layout."""
    if work_root is not None and data_root is not None:
        raise ValueError("data_root and work_root are mutually exclusive")
    if data_root is not None:
        return data_root.resolve() / "runs"
    return resolve_work_root(work_root)


def stage_checkpoint_root(repo: Path, data_root: Path | None, work: Path) -> Path:
    # Historical API callers keep the checkpoint location used by their archive.
    return (repo if data_root is not None else work) / "checkpoints/grpo"


def collect_training_metrics(
    *,
    repo: Path,
    data_root: Path | None = None,
    work_root: Path | None = None,
    run_name: str,
    after_step: int,
    step: int,
    log_files: list[Path],
    metrics_files: list[Path] | None = None,
    recover_offline: bool = True,
) -> tuple[dict, dict, list]:
    work = stage_work_root(data_root, work_root)
    records, warnings = {}, []
    sources = {"logs": [], "structured_metrics": [], "wandb_offline": []}
    for log in log_files:
        if not log.is_file():
            warnings.append(f"未找到阶段日志：{log}")
            continue
        rejected: list[str] = []
        parsed, conflicts = parse_metrics(
            log.read_text(encoding="utf-8-sig", errors="replace"), rejected=rejected
        )
        warnings.extend(f"{log}: {note}" for note in rejected)
        conflicts += merge_metrics(records, parsed)
        sources["logs"].append(str(log))
        if conflicts:
            warnings.append(f"日志同 step 指标冲突，保留后值：{sorted(set(conflicts))}")
    paths = (
        metrics_files
        if metrics_files is not None
        else [stage_checkpoint_root(repo, data_root, work) / run_name / "metrics.jsonl"]
    )
    # Durable trainer records are authoritative; console remains a fallback.
    for path in paths:
        if path.is_file():
            parsed, notes = read_metrics(path)
            warnings.extend(notes)
            merge_metrics(records, parsed)
            sources["structured_metrics"].append(str(path))
    if recover_offline and not metric_coverage(records, after_step, step)["complete"]:
        recovered, paths, notes = recover_wandb(
            work / "wandb/wandb", run_name, after_step, step
        )
        warnings.extend(notes)
        # Fill gaps from W&B without overwriting the direct journal/console.
        merge_metrics(recovered, records)
        records = recovered
        sources["wandb_offline"] = paths
    return records, sources, warnings


def analyze(
    *,
    repo: Path,
    data_root: Path | None = None,
    work_root: Path | None = None,
    run_name: str,
    after_step: int,
    step: int,
    log_files: list[Path],
    output: Path,
    metrics_files: list[Path] | None = None,
    run_dir: Path | None = None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("invalid run name")
    if not 0 <= after_step < step:
        raise ValueError("require 0 <= after-step < step")
    work = stage_work_root(data_root, work_root)
    if metrics_files is None and run_dir is not None:
        metrics_files = [run_dir / "metrics.jsonl"]
    all_metrics, metric_sources, warnings = collect_training_metrics(
        repo=repo,
        data_root=data_root,
        work_root=work_root,
        run_name=run_name,
        after_step=after_step,
        step=step,
        log_files=log_files,
        metrics_files=metrics_files,
    )
    coverage = metric_coverage(all_metrics, after_step, step)
    if coverage["missing_by_metric"]:
        warnings.append(f"关键训练指标缺失：{coverage['missing_by_metric']}")
    metrics = [
        {"step": number, **values}
        for number, values in sorted(all_metrics.items())
        if after_step < number <= step and "training/global_step" in values
    ]
    present = {r["step"] for r in metrics}
    missing = sorted(set(range(after_step + 1, step + 1)) - present)
    if missing:
        warnings.append(
            f"缺少训练 step 指标：{missing}；不能据此确认完整阶段的稳定性。"
        )
    nonfinite = [
        (r["step"], k) for r in metrics for k, v in r.items() if not math.isfinite(v)
    ]
    if nonfinite:
        warnings.append(f"发现非有限指标：{nonfinite}")
    failed = [
        r["step"] for r in metrics if r.get("actor/optimizer_update_success", 1) != 1
    ]
    if failed:
        warnings.append(f"optimizer 更新未成功：{failed}")
    if any("actor/optimizer_update_success" not in r for r in metrics):
        warnings.append(
            "部分 step 没有 optimizer 成功指标，无法检查这些更新的成功状态。"
        )
    run_dir = run_dir or stage_checkpoint_root(repo, data_root, work) / run_name
    generation_dir = work / "grpo" / run_name
    baseline = load_rows(generation_dir / "validation/0.jsonl", 0)
    current = load_rows(generation_dir / f"validation/{step}.jsonl", step)
    comparison, pairs, changed = compare_validation(baseline, current)
    verify_receipts(run_dir, step, comparison)
    rollouts, missing_rollouts = [], []
    for number in range(after_step + 1, step + 1):
        path = generation_dir / f"rollouts/{number}.jsonl"
        if not path.is_file():
            missing_rollouts.append(number)
            continue
        result = row_summary(load_rows(path, number))
        console = all_metrics.get(number, {})
        expected = console.get("trace/dynamic/accepted_rollouts")
        if expected is not None and result["rows"] != expected:
            warnings.append(f"step {number} 已接收轨迹数与日志不一致。")
        expected_acc = console.get("trace/dynamic/accepted_acc_mean")
        if expected_acc is not None and not math.isclose(
            result["pass_at_1"], expected_acc, abs_tol=1e-12, rel_tol=0
        ):
            warnings.append(f"step {number} 已接收轨迹正确率与日志不一致。")
        result["accepted_acc_mean"] = result.pop("pass_at_1")
        rollouts.append(
            {
                "step": number,
                **{k: v for k, v in result.items() if k != "termination_counts"},
                **{
                    f"termination/{k}": v / result["rows"]
                    for k, v in result["termination_counts"].items()
                },
            }
        )
    if missing_rollouts:
        warnings.append(f"缺少已接收训练轨迹 JSONL：{missing_rollouts}")
    trends = {}
    width = min(10, max(1, (step - after_step) // 2))
    for key in WATCH:
        groups = (
            range(after_step + 1, after_step + width + 1),
            range(step - width + 1, step + 1),
        )
        values = [
            [
                r[key]
                for r in metrics
                if r["step"] in group and key in r and math.isfinite(r[key])
            ]
            for group in groups
        ]
        trends[key] = {
            "first_mean": mean(values[0]) if values[0] else None,
            "last_mean": mean(values[1]) if values[1] else None,
            "first_n": len(values[0]),
            "last_n": len(values[1]),
        }
    summary = {
        "run_name": run_name,
        "after_step": after_step,
        "step": step,
        "sources": {
            "logs": [str(p) for p in log_files],
            "metrics": metric_sources,
            "checkpoint_run_dir": str(run_dir),
            "generation_dir": str(generation_dir),
        },
        "training_steps_found": len(metrics),
        "metric_coverage": coverage,
        "python_error_penalty": coverage["python_error_penalty"],
        "missing_training_steps": missing,
        "missing_rollout_steps": missing_rollouts,
        "trend_window": width,
        "trends": trends,
        "validation": comparison,
        "warnings": warnings,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "training_metrics.csv", metrics, ("step",))
    write_csv(output / "training_rollouts.csv", rollouts, ("step",))
    write_csv(
        output / "validation_comparison.csv", pairs, ("question_key", "transition")
    )
    write_jsonl(output / "changed_validation_questions.jsonl", changed)
    # Preserve the longest incorrect answers for manual loop inspection.
    write_jsonl(
        output / "longest_incorrect_validation.jsonl",
        sorted(
            [r for r in current if r["score"] == 0],
            key=lambda r: len(r["output"]),
            reverse=True,
        )[:10],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def format_number(value):
        return "缺失" if value is None else f"{value:.6g}"

    b, c = comparison["baseline"], comparison["current"]
    lines = [
        f"# {run_name}：step {after_step + 1}–{step} 分析",
        "",
        f"训练指标：{len(metrics)}/{step - after_step} step；"
        f"已接收轨迹文件：{len(rollouts)}/{step - after_step}。",
        "Python 错误信用诊断："
        + (
            "不可用（历史数据未记录，不视为零）"
            if coverage["python_error_penalty"]["availability"] == "unavailable"
            else "已记录；启用 "
            f"{len(coverage['python_error_penalty']['enabled_steps'])} step"
        ),
        "",
        f"固定验证集配对：{b['rows']} 题；baseline {b['correct']}/{b['rows']} "
        f"({b['pass_at_1']:.2%})；step {step} {c['correct']}/{c['rows']} "
        f"({c['pass_at_1']:.2%})；"
        f"差值 {comparison['delta_percentage_points']:+.4f} 个百分点。",
        "",
        f"逐题变化：{comparison['transitions']}。"
        f"配对 McNemar 双侧精确 p={comparison['paired_mcnemar_exact_p']:.6g}。",
        "这个 p 值只描述当前固定集合的配对差异，不代表整体数学能力或后续训练稳定性。",
        "",
        "## 验证诊断均值",
        "",
        "| 指标 | baseline | 当前 |",
        "|---|---:|---:|",
    ]
    for key in (
        *DIAGNOSTICS,
        "output_chars_mean",
        "output_chars_p90",
        "output_chars_incorrect_mean",
    ):
        lines.append(f"| `{key}` | {format_number(b[key])} | {format_number(c[key])} |")
    lines += [
        "",
        f"验证结束原因：baseline {b['termination_counts']}；"
        f"当前 {c['termination_counts']}。",
        "",
        f"## 训练趋势（前 {width} / 后 {width} 个 step 的均值）",
        "",
        "| 指标 | 前期 | 后期 | 有效 step 数（前/后） |",
        "|---|---:|---:|---:|",
    ]
    for key, trend in trends.items():
        lines.append(
            f"| `{key}` | {format_number(trend['first_mean'])} "
            f"| {format_number(trend['last_mean'])} "
            f"| {trend['first_n']}/{trend['last_n']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- candidate_acc_mean 是筛选前候选正确率；"
        "accepted_acc_mean 和训练 JSONL 都来自筛选后批次。"
        "候选题也随采样变化，能力比较以固定验证集为准。",
        "- CSV 保存全部控制台标量，可按 step 画曲线；"
        "重点联合查看正确率、响应长度/截断、非法动作、reference KL、梯度和熵，"
        "单个均值不能排除重复退化。",
        "- output_chars 是解码文本字符数（含工具反馈），不是 token 数；"
        "训练 response_length 的 token 指标来自日志。",
        "- changed_validation_questions.jsonl 保存变对/变错题的两次完整输出；"
        "longest_incorrect_validation.jsonl 保存当前最长的 10 条错误输出，"
        "供人工检查重复或协议问题。",
        "- 缺失字段显示为缺失，不当作 0；此报告不会自动宣称训练稳定。",
        "",
        "## 数据检查",
        "",
    ]
    lines += [f"- {warning}" for warning in warnings] or [
        "- 未发现指标缺步或上述数据检查异常。"
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--after-step", type=int, default=0)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--data-root", type=Path, help="historical DATA/runs layout")
    roots.add_argument("--work-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--log-file",
        type=Path,
        action="append",
        help="repeat for multiple logs; default logs/<run>-stage-<after>.log",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics-file", type=Path, action="append")
    args = parser.parse_args()
    configure_asset_environment(args.assets_root, args.work_root)
    work = stage_work_root(args.data_root, args.work_root)
    output = args.output_dir or work / "analysis" / args.run_name / f"step-{args.step}"
    logs = args.log_file or [
        work / "logs" / f"{args.run_name}-stage-{args.after_step}.log"
    ]
    try:
        summary = analyze(
            repo=args.project_root,
            data_root=args.data_root,
            work_root=args.work_root,
            run_name=args.run_name,
            after_step=args.after_step,
            step=args.step,
            log_files=logs,
            output=output,
            metrics_files=args.metrics_file,
            run_dir=args.run_dir,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    v = summary["validation"]
    print(
        f"Training metrics: {summary['training_steps_found']}/"
        f"{args.step - args.after_step} steps"
    )
    print(
        f"Baseline: {v['baseline']['correct']}/{v['baseline']['rows']} "
        f"({v['baseline']['pass_at_1']:.2%})"
    )
    print(
        f"Step {args.step}: {v['current']['correct']}/{v['current']['rows']} "
        f"({v['current']['pass_at_1']:.2%})"
    )
    print(
        f"Delta: {v['delta_percentage_points']:+.4f} percentage points; "
        f"transitions: {v['transitions']}"
    )
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    print(f"Report: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
