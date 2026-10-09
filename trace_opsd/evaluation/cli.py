"""Command-line entry points for the independent mathematics evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from trace_opsd.assets import asset_path
from trace_opsd.compat import read_json

from .data import BY_NAME, prepare
from .runner import EvalConfig
from .workflow import report, run_evaluation, score_run


def _names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)) or set(names) - BY_NAME.keys():
        raise argparse.ArgumentTypeError(
            f"benchmarks must be distinct names from {','.join(BY_NAME)}"
        )
    return names


def _pass_ks(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pass-ks must be comma-separated positive integers"
        ) from exc
    if not numbers or any(k < 1 for k in numbers) or len(set(numbers)) != len(numbers):
        raise argparse.ArgumentTypeError(
            "pass-ks must contain distinct positive integers"
        )
    return numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-opsd-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="download and seal official dataset revisions"
    )
    prep.add_argument("--dataset-dir", type=Path, default=Path("data/evaluation"))
    prep.add_argument("--benchmarks", type=_names, default=list(BY_NAME))
    prep.add_argument("--train-parquet", type=Path, action="append", default=[])

    run = sub.add_parser("run", help="generate and score one model on a fixed snapshot")
    run.add_argument("--dataset-dir", type=Path, default=Path("data/evaluation"))
    run.add_argument("--benchmarks", type=_names, default=list(BY_NAME))
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--served-model", default="trace-opsd-eval")
    run.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--tool-url", default="http://127.0.0.1:8002")
    run.add_argument(
        "--index-path",
        type=Path,
        default=asset_path("retrieval", "general_search", "index", "e5_Flat.index"),
    )
    run.add_argument(
        "--corpus-path",
        type=Path,
        default=asset_path(
            "retrieval", "general_search", "corpus", "search_r1_corpus.jsonl"
        ),
    )
    run.add_argument("--mode", choices=("tool", "direct"), default="tool")
    run.add_argument("--thinking-mode", choices=("auto", "on", "off"), default="auto")
    run.add_argument("--limit-per-benchmark", type=int)
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument(
        "--execution-metadata",
        type=Path,
        help="launcher JSON recording inference settings and backend versions",
    )

    score = sub.add_parser(
        "score", help="rescore saved trajectories without generation"
    )
    score.add_argument("--dataset-dir", type=Path, default=Path("data/evaluation"))
    score.add_argument("--run-dir", type=Path, required=True)

    summary = sub.add_parser("report", help="combine only protocol-identical full runs")
    summary.add_argument("--run-dir", type=Path, action="append", required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--baseline-run-dir", type=Path)

    sampled = sub.add_parser(
        "sample",
        help="generate independent stochastic samples and score Pass/Avg/Majority",
    )
    sampled.add_argument("--dataset-dir", type=Path, default=Path("data/evaluation"))
    sampled.add_argument("--benchmarks", type=_names, default=list(BY_NAME))
    sampled.add_argument("--model-path", type=Path, required=True)
    sampled.add_argument("--run-dir", type=Path, required=True)
    sampled.add_argument("--served-model", default="trace-opsd-eval")
    sampled.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    sampled.add_argument("--tool-url", default="http://127.0.0.1:8002")
    sampled.add_argument(
        "--index-path",
        type=Path,
        default=asset_path("retrieval", "general_search", "index", "e5_Flat.index"),
    )
    sampled.add_argument(
        "--corpus-path",
        type=Path,
        default=asset_path(
            "retrieval", "general_search", "corpus", "search_r1_corpus.jsonl"
        ),
    )
    sampled.add_argument("--mode", choices=("tool", "direct"), default="tool")
    sampled.add_argument(
        "--thinking-mode", choices=("auto", "on", "off"), default="auto"
    )
    sampled.add_argument("--limit-per-benchmark", type=int)
    sampled.add_argument("--concurrency", type=int, default=16)
    sampled.add_argument("--execution-metadata", type=Path)
    sampled.add_argument("--samples-per-question", "--n-samples", type=int, default=8)
    sampled.add_argument(
        "--pass-ks",
        type=_pass_ks,
        help="defaults to 1,2,4,8 up to sample count, plus the sample count",
    )
    sampled.add_argument("--temperature", type=float, default=0.7)
    sampled.add_argument("--top-p", type=float, default=0.95)
    sampled.add_argument("--top-k", type=int, default=-1)
    sampled.add_argument("--seed", type=int, default=42)
    sampled.add_argument("--context-tokens", type=int, default=8192)
    sampled.add_argument("--total-response-tokens", type=int, default=4096)
    sampled.add_argument("--max-tokens-per-turn", type=int, default=1024)
    sampled.add_argument("--max-observation-tokens", type=int, default=512)
    sampled.add_argument("--max-tool-calls", type=int, default=5)
    sampled.add_argument("--max-consecutive-invalid-actions", type=int, default=2)

    sampled_score = sub.add_parser(
        "sample-score", help="score and vote stored samples on the main CPU thread"
    )
    sampled_score.add_argument(
        "--dataset-dir", type=Path, default=Path("data/evaluation")
    )
    sampled_score.add_argument("--run-dir", type=Path, required=True)
    sampled_report = sub.add_parser(
        "sample-report", help="combine complete protocol-identical sampled runs"
    )
    sampled_report.add_argument("--run-dir", type=Path, action="append", required=True)
    sampled_report.add_argument("--output-dir", type=Path, required=True)
    sampled_report.add_argument("--baseline-run-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        train = args.train_parquet
        default = asset_path("datasets", "numinamath", "rl_trace_opsd.parquet")
        if not train and default.is_file():
            train = [default]
        result = prepare(args.dataset_dir, names=args.benchmarks, train_paths=train)
    elif args.command == "run":
        result = asyncio.run(
            run_evaluation(
                run_dir=args.run_dir,
                dataset_dir=args.dataset_dir,
                benchmarks=args.benchmarks,
                model_path=args.model_path,
                served_model=args.served_model,
                model_url=args.model_url,
                tool_url=args.tool_url if args.mode == "tool" else None,
                index_path=args.index_path if args.mode == "tool" else None,
                corpus_path=args.corpus_path if args.mode == "tool" else None,
                mode=args.mode,
                thinking_mode=args.thinking_mode,
                limit_per_benchmark=args.limit_per_benchmark,
                concurrency=args.concurrency,
                execution_metadata=(
                    read_json(args.execution_metadata)
                    if args.execution_metadata
                    else None
                ),
            )
        )
    elif args.command == "score":
        result = score_run(args.run_dir, args.dataset_dir)
    elif args.command == "report":
        result = report(
            args.run_dir, args.output_dir, baseline_run_dir=args.baseline_run_dir
        )
    elif args.command == "sample":
        from .sampling_workflow import SAMPLED_PROTOCOL, run_sampled_evaluation

        config = EvalConfig(
            protocol=SAMPLED_PROTOCOL,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            context_tokens=args.context_tokens,
            total_response_tokens=args.total_response_tokens,
            max_tokens_per_turn=args.max_tokens_per_turn,
            max_observation_tokens=args.max_observation_tokens,
            max_tool_calls=args.max_tool_calls,
            max_consecutive_invalid_actions=args.max_consecutive_invalid_actions,
        )
        result = asyncio.run(
            run_sampled_evaluation(
                run_dir=args.run_dir,
                dataset_dir=args.dataset_dir,
                benchmarks=args.benchmarks,
                model_path=args.model_path,
                served_model=args.served_model,
                model_url=args.model_url,
                tool_url=args.tool_url if args.mode == "tool" else None,
                index_path=args.index_path if args.mode == "tool" else None,
                corpus_path=args.corpus_path if args.mode == "tool" else None,
                mode=args.mode,
                thinking_mode=args.thinking_mode,
                limit_per_benchmark=args.limit_per_benchmark,
                concurrency=args.concurrency,
                config=config,
                n_samples=args.samples_per_question,
                pass_ks=args.pass_ks
                or tuple(
                    sorted(
                        {
                            *(
                                k
                                for k in (1, 2, 4, 8)
                                if k <= args.samples_per_question
                            ),
                            args.samples_per_question,
                        }
                    )
                ),
                execution_metadata=read_json(args.execution_metadata)
                if args.execution_metadata
                else None,
            )
        )
    elif args.command == "sample-score":
        from .sampling_workflow import score_sampled_run

        result = score_sampled_run(args.run_dir, args.dataset_dir)
    else:
        from .sampling_workflow import report_sampled_runs

        result = report_sampled_runs(
            args.run_dir, args.output_dir, baseline_run_dir=args.baseline_run_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"sample", "sample-score"} and not result.get(
        "official_metrics_available"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
