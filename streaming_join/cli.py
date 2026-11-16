"""Command-line entry point for the streaming full-outer similarity join.

Reads two JSONL files (one JSON object per line), joins them with the
:mod:`simjoin` engine, and writes one JSON result per line to stdout.
The left file is read lazily line by line; outputs are flushed per row,
so the command can be used in shell pipelines without either input (or
the result) fitting in memory.  Only the right side is indexed in
memory, as required by the MinHashLSH algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

# Self-contained under `python -I`: isolated mode does not prepend the
# script directory to sys.path, so add it explicitly before importing the
# sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simjoin import streaming_full_outer_similarity_join  # noqa: E402


def _read_jsonl(path: str) -> Iterator[dict]:
    if path == "-":
        stream = sys.stdin
        owns = False
    else:
        stream = open(path, "r", encoding="utf-8")
        owns = True
    try:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path} line {line_no}: {exc}") from exc
    finally:
        if owns:
            stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Streaming full-outer similarity join of two JSONL record batches."
    )
    parser.add_argument("left", help="left JSONL file (streamed row by row), '-' for stdin")
    parser.add_argument("right", help="right JSONL file (fully indexed into MinHashLSH)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="minimum exact Jaccard similarity to confirm a pair (default: 0.8)",
    )
    parser.add_argument(
        "--token-field",
        default="tokens",
        help="record field containing the token list/set (default: tokens)",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="number of MinHash permutation functions (default: 128)",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0.0, 1.0]")

    right = list(_read_jsonl(args.right))
    join = streaming_full_outer_similarity_join(
        _read_jsonl(args.left),
        right,
        threshold=args.threshold,
        token_field=args.token_field,
        num_perm=args.num_perm,
    )
    try:
        for result in join:
            sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            sys.stdout.flush()
    except BrokenPipeError:
        # Downstream consumer (e.g. `head`) closed early; stop quietly.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
