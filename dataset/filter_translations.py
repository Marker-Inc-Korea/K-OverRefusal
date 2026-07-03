"""Filter an already-translated jsonl with a separate (filter) model.

Two-stage translation pipeline: a dedicated translation model (e.g.
nayohan/llama3-instrucTrans-enko-8b, which rarely refuses) produces the Korean
translations via `translate_dataset.py --translate_only`, then this script hands the
quality check to a stronger judge model (e.g. gpt-oss-120b).

Reuses the heuristics + second-pass LLM check from translate_dataset.apply_translation_filter,
so the filtering behavior is identical to the single-model path. Writes clean rows to
--save_path and flagged rows to <save_path>_rejected.jsonl; each record gets
translation_ok (bool) and translation_flags (list).
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

from datasets import Dataset

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from util import build_engine

try:
    from translate_dataset import apply_translation_filter
except ImportError:
    from dataset.translate_dataset import apply_translation_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--filter_model_engine_backend", type=str, default="vllm")
    parser.add_argument("--filter_model_backend_base_url", type=str, default=None)
    parser.add_argument("--filter_model_num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--skip_mm_profiling", action="store_true")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Translated jsonl from translate_dataset.py --translate_only.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Where clean rows are written; rejects go to <save_path>_rejected.jsonl.")
    parser.add_argument("--source_column", type=str, default="prompt",
                        help="Column holding the original English text.")
    parser.add_argument("--translation_column", type=str, default="prompt_ko",
                        help="Column holding the Korean translation to check.")

    parser.add_argument("--filter_batch_size", type=int, default=32)
    parser.add_argument("--filter_max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    examples: List[Dict] = [dict(ex) for ex in Dataset.from_json(args.data_path)]
    print(f"Loaded {len(examples)} translated rows from {args.data_path}")

    missing = [c for c in (args.source_column, args.translation_column) if c not in (examples[0] if examples else {})]
    if missing:
        raise ValueError(f"Columns {missing} not found in {args.data_path}. "
                         f"Available: {list(examples[0].keys()) if examples else []}")

    sources = [str(ex.get(args.source_column) or "") for ex in examples]
    translations = [str(ex.get(args.translation_column) or "") for ex in examples]

    model = build_engine(
        args.filter_model, args.filter_model_engine_backend,
        num_gpus=args.filter_model_num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        base_url=args.filter_model_backend_base_url,
        skip_mm_profiling=args.skip_mm_profiling,
    )

    apply_translation_filter(
        model, examples, sources, translations, args.save_path,
        skip_llm_filter=False,
        filter_batch_size=args.filter_batch_size,
        filter_max_new_tokens=args.filter_max_new_tokens,
        seed=args.seed,
        translation_column=args.translation_column,
    )


if __name__ == "__main__":
    main()
