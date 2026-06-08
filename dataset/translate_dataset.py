import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine
try:
    from generation_util import XSTEST_TRANSLATION_PROMPT
except ImportError:
    from dataset.generation_util import XSTEST_TRANSLATION_PROMPT


DEFAULT_SAVE_PATH = "/data/home/mk05/FR/dataset/xstest/xstest_ko.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)

    parser.add_argument("--dataset_name", type=str, default="walledai/XSTest")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--prompt_column", type=str, default="prompt")

    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_num_examples", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning", type=str, default="low")
    return parser.parse_args()


def load_examples(args) -> List[Dict]:
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.max_num_examples is not None:
        dataset = dataset.select(range(min(args.max_num_examples, len(dataset))))

    if args.prompt_column not in dataset.column_names:
        raise ValueError(f"Column '{args.prompt_column}' not found. Available: {dataset.column_names}")

    return [dict(example) for example in tqdm(dataset, desc="Loading examples")]


def build_backend_kwargs(args) -> Dict:
    if args.model_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.model_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.max_model_len is not None:
            backend_kwargs["max_model_len"] = args.max_model_len
        if args.max_num_seqs is not None:
            backend_kwargs["max_num_seqs"] = args.max_num_seqs
        return backend_kwargs

    if args.model_engine_backend in {"vllm-openai", "openrouter"}:
        backend_kwargs = {}
        if args.model_backend_base_url is not None:
            backend_kwargs["base_url"] = args.model_backend_base_url
        return backend_kwargs

    return {}


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


_PUNCT = r"[.?!？。！]"


def _last_sentence_in_quotes(text: str) -> Optional[str]:
    """Last backtick- or double-quote-delimited sentence ending with punctuation."""
    for pattern in (rf"`([^`]{{5,}}{_PUNCT})`", rf'"([^"]{{5,}}{_PUNCT})"'):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def parse_translation(text: str) -> str:
    text = str(text or "").strip()
    # Last start-of-line [OUTPUT]: match — avoids matching format descriptions in thinking
    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate
    # Fallback: last quoted sentence (backtick or double-quote) — handles truncation
    quoted = _last_sentence_in_quotes(text)
    if quoted:
        return quoted
    return text


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    ensure_parent_dir(args.save_path)

    examples = load_examples(args)
    print(f"Loaded {len(examples)} examples from {args.dataset_name} ({args.dataset_split} split)")

    backend_kwargs = build_backend_kwargs(args)
    model = VLMInferenceEngine(
        args.model,
        backend=args.model_engine_backend,
        backend_kwargs=backend_kwargs,
    )
    gen_params = UniversalGenParams(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        reasoning=args.reasoning,
    )

    prompts = [
        XSTEST_TRANSLATION_PROMPT.format(text=str(ex.get(args.prompt_column) or ""))
        for ex in examples
    ]

    translations: List[str] = []
    for batch in tqdm(list(batched(prompts, args.batch_size)), desc="Translating"):
        gen_args = GenerationArgs(
            engine_input=batch,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        for output in outputs:
            raw = output.output_seqs[0] if output.output_seqs else ""
            translations.append(parse_translation(raw))

    failed = 0
    with open(args.save_path, "w", encoding="utf-8") as f:
        for example, translation in zip(examples, translations):
            if not translation:
                failed += 1
            record = dict(example)
            record["prompt_ko"] = translation
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(examples)} records to {args.save_path} ({failed} empty translations)")


if __name__ == "__main__":
    main()
