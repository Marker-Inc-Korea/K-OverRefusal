"""Recompute metrics.json from an existing evaluation_results.jsonl without re-running inference."""

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

try:
    from eval.eval_util import compute_metrics, compute_exaggerate_safety_metrics, dump_metrics
except ImportError:
    from eval_util import compute_metrics, compute_exaggerate_safety_metrics, dump_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute metrics.json from evaluation_results.jsonl"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to evaluation_results.jsonl",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save metrics.json. Defaults to the directory of results_path.",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="label",
        help="Column name for the query type label (safe/unsafe). "
             "If absent in the file, compute_metrics (no label) is used instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = args.save_dir or os.path.dirname(os.path.abspath(args.results_path))
    os.makedirs(save_dir, exist_ok=True)

    dataset = Dataset.from_json(args.results_path)
    print(f"Loaded {len(dataset)} examples from {args.results_path}")

    if args.label_key in dataset.column_names:
        metrics = compute_exaggerate_safety_metrics(dataset, example_type_key=args.label_key)
    else:
        print(f"Column '{args.label_key}' not found — falling back to compute_metrics (no safe/unsafe split).")
        metrics = compute_metrics(dataset)

    metrics_str = dump_metrics(metrics)
    metrics_save_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write(metrics_str + "\n")

    print(f"Saved metrics → {metrics_save_path}")
    print(metrics_str)


if __name__ == "__main__":
    main()
