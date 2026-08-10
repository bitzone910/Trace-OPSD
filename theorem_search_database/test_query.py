"""Send one query to a running Theorem Search Database service."""

from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Theorem Search Database.")
    parser.add_argument("--query", default="When is a module flat?")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--url", default="http://127.0.0.1:8001/retrieve")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no_scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.query.strip():
        raise ValueError("query must not be empty")
    if args.topk <= 0:
        raise ValueError("topk must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    response = requests.post(
        args.url,
        json={
            "queries": [args.query],
            "topk": args.topk,
            "return_scores": not args.no_scores,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
