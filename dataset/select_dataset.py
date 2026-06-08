import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine
try:
    from generation_util import (
        SEED_INSTRUCTION_SELECTION_PROMPT,
        SEED_INSTRUCTION_GENERATION_PROMPT,
    )
except ImportError:
    from dataset.generation_util import (
        SEED_INSTRUCTION_SELECTION_PROMPT,
        SEED_INSTRUCTION_GENERATION_PROMPT,
    )


DEFAULT_SEED_WORD_PATH = "/data/home/mk05/FR/dataset/seed_word.csv"
DEFAULT_SAVE_PATH = "/data/home/mk05/FR/dataset/test/semantic_match_dataset.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)

    parser.add_argument("--dataset_name", type=str, default="iknow-lab/wildguardmix-test-ko")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--instruction_column", type=str, default="prompt_ko")
    parser.add_argument("--response_column", type=str, default="response_ko")

    parser.add_argument("--seed_word_path", type=str, default=DEFAULT_SEED_WORD_PATH)
    parser.add_argument("--pseudo_keyword_col", type=int, default=0)
    parser.add_argument("--pseudo_instruction_col", type=int, default=1)
    parser.add_argument("--harmful_keyword_col", type=int, default=2)

    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--log_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--max_num_seeds", type=int, default=None)
    parser.add_argument("--start_seed_index", type=int, default=0)
    parser.add_argument("--append", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning", type=str, default="low")
    return parser.parse_args()


def read_seed_rows(
    path: str,
    pseudo_keyword_col: int,
    pseudo_instruction_col: int,
    harmful_keyword_col: int,
) -> List[Dict[str, str]]:
    rows = []
    seen = set()
    max_col = max(pseudo_keyword_col, pseudo_instruction_col, harmful_keyword_col)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= max_col:
                continue

            pseudo_keyword = row[pseudo_keyword_col].strip()
            pseudo_instruction = row[pseudo_instruction_col].strip()
            harmful_keyword = row[harmful_keyword_col].strip()

            if not harmful_keyword or harmful_keyword.lower() in {"harmful_keyword", "target", "keyword"}:
                continue

            key = (pseudo_instruction, harmful_keyword)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "pseudo_keyword": pseudo_keyword,
                "pseudo_instruction": pseudo_instruction,
                "harmful_keyword": harmful_keyword,
            })

    return rows


def load_candidate_examples(args) -> List[Dict[str, str]]:
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.filter(
        lambda x: str(x.get("response_harm_label", "")).strip().lower() == "harmful"
    )
    if args.max_num_examples is not None:
        dataset = dataset.select(range(min(args.max_num_examples, len(dataset))))

    missing_columns = [
        c for c in [args.instruction_column, args.response_column]
        if c not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    examples = []
    for example in tqdm(dataset, desc="Loading candidate examples"):
        instruction = str(example.get(args.instruction_column) or "").strip()
        if not instruction:
            continue
        examples.append({
            "instruction": instruction,
            "response": str(example.get(args.response_column) or ""),
        })
    return examples


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


def keyword_filter(examples: List[Dict[str, str]], keyword: str) -> List[Dict[str, str]]:
    return [ex for ex in examples if keyword in ex["instruction"]]


def build_selection_prompt(pseudo_instruction: str, candidates: List[Dict[str, str]]) -> str:
    candidates_text = "\n".join(
        f"[{i}] {ex['instruction']}" for i, ex in enumerate(candidates)
    )
    return SEED_INSTRUCTION_SELECTION_PROMPT.format(
        pseudo_instruction=pseudo_instruction,
        candidates=candidates_text,
    )


def build_generation_prompt(pseudo_instruction: str, harmful_keyword: str) -> str:
    return SEED_INSTRUCTION_GENERATION_PROMPT.format(
        pseudo_instruction=pseudo_instruction,
        harmful_keyword=harmful_keyword,
    )


_PUNCT = r"[.?!？。！]"


def _last_sentence_in_quotes(text: str) -> Optional[str]:
    """Last backtick- or double-quote-delimited sentence ending with punctuation."""
    for pattern in (rf"`([^`]{{5,}}{_PUNCT})`", rf'"([^"]{{5,}}{_PUNCT})"'):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def _parse_output_tag(text: str) -> Optional[str]:
    """
    Return the last start-of-line [OUTPUT]: match, skipping template placeholders.
    Using re.MULTILINE so ^ anchors to each line start, which avoids matching
    [OUTPUT]: references inside indented thinking or backtick code spans.
    """
    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate
    return None


def parse_selection_output(text: str, num_candidates: int) -> int:
    text = str(text or "").strip()
    # Last start-of-line [OUTPUT]: N
    matches = re.findall(r"^\[OUTPUT\]:\s*(\d+)", text, re.MULTILINE)
    for m in reversed(matches):
        idx = int(m)
        if 0 <= idx < num_candidates:
            return idx
    # Fallback: last valid number anywhere in text
    for n in reversed(re.findall(r"\d+", text)):
        idx = int(n)
        if 0 <= idx < num_candidates:
            return idx
    return 0


def parse_generation_output(text: str) -> Optional[str]:
    text = str(text or "").strip()
    result = _parse_output_tag(text)
    if result:
        return result
    # Fallback: last quoted sentence (backtick or double-quote) — handles truncation
    quoted = _last_sentence_in_quotes(text)
    if quoted:
        return quoted
    return text or None


def make_record(seed_row: Dict, instruction: str) -> Dict:
    return {
        "output": "",
        "input": "",
        "instruction": instruction,
        "response": "",
        "seed": seed_row["pseudo_keyword"],
    }


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def append_jsonl(path: Optional[str], records: List[Dict]):
    if not records or path is None:
        return
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def initialize_output_files(args):
    for path in [args.save_path, args.log_path]:
        if path is None:
            continue
        ensure_parent_dir(path)
        if not args.append:
            with open(path, "w", encoding="utf-8"):
                pass


def run_batch_generate(model, prompts: List[str], gen_params, batch_size: int) -> List[str]:
    results = []
    for batch in tqdm(list(batched(prompts, batch_size)), desc="  LLM batch"):
        gen_args = GenerationArgs(
            engine_input=batch,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        results.extend(
            output.output_seqs[0] if output.output_seqs else "" for output in outputs
        )
    return results


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    initialize_output_files(args)

    seed_rows = read_seed_rows(
        args.seed_word_path,
        args.pseudo_keyword_col,
        args.pseudo_instruction_col,
        args.harmful_keyword_col,
    )
    seed_rows = seed_rows[args.start_seed_index:]
    if args.max_num_seeds is not None:
        seed_rows = seed_rows[: args.max_num_seeds]

    examples = load_candidate_examples(args)
    print(f"Loaded {len(examples)} candidate examples, {len(seed_rows)} seeds")

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

    # Step 1: keyword matching — categorize all seeds
    direct_seeds: List[Tuple[Dict, Dict]] = []       # (seed_row, single_candidate)
    selection_seeds: List[Tuple[Dict, List]] = []    # (seed_row, candidates)
    generation_seeds: List[Dict] = []                # seed_row (no candidates found)

    print("Step 1: Keyword matching...")
    for seed_row in tqdm(seed_rows, desc="Filtering"):
        candidates = keyword_filter(examples, seed_row["harmful_keyword"])

        if len(candidates) == 0:
            generation_seeds.append(seed_row)
        elif len(candidates) == 1:
            direct_seeds.append((seed_row, candidates[0]))
        else:
            if len(candidates) > args.max_candidates:
                candidates = random.sample(candidates, args.max_candidates)
            selection_seeds.append((seed_row, candidates))

    print(
        f"  direct={len(direct_seeds)}, "
        f"needs_selection={len(selection_seeds)}, "
        f"needs_generation={len(generation_seeds)}"
    )

    total_saved = 0

    # Direct: single candidate, no LLM needed
    if direct_seeds:
        records = [make_record(sr, cand["instruction"]) for sr, cand in direct_seeds]
        append_jsonl(args.save_path, records)
        total_saved += len(records)
        print(f"  Saved {len(records)} direct records")

    # Step 2: LLM selection — multiple candidates, pick best
    if selection_seeds:
        print(f"Step 2: LLM selection for {len(selection_seeds)} seeds...")
        prompts = [
            build_selection_prompt(sr["pseudo_instruction"], cands)
            for sr, cands in selection_seeds
        ]
        raw_outputs = run_batch_generate(model, prompts, gen_params, args.batch_size)

        selection_records = []
        log_records = []
        for (seed_row, candidates), raw_output in zip(selection_seeds, raw_outputs):
            idx = parse_selection_output(raw_output, len(candidates))
            selection_records.append(make_record(seed_row, candidates[idx]["instruction"]))
            if args.log_path:
                log_records.append({
                    "step": "selection",
                    "seed_harmful_keyword": seed_row["harmful_keyword"],
                    "pseudo_instruction": seed_row["pseudo_instruction"],
                    "num_candidates": len(candidates),
                    "selected_index": idx,
                    "selected_instruction": candidates[idx]["instruction"],
                    "raw_output": raw_output,
                })

        append_jsonl(args.save_path, selection_records)
        append_jsonl(args.log_path, log_records)
        total_saved += len(selection_records)
        print(f"  Saved {len(selection_records)} selection records")

    # Step 3: LLM generation — no candidates found, generate from scratch
    if generation_seeds:
        print(f"Step 3: LLM generation for {len(generation_seeds)} seeds...")
        prompts = [
            build_generation_prompt(sr["pseudo_instruction"], sr["harmful_keyword"])
            for sr in generation_seeds
        ]
        raw_outputs = run_batch_generate(model, prompts, gen_params, args.batch_size)

        generation_records = []
        log_records = []
        failed = 0
        for seed_row, raw_output in zip(generation_seeds, raw_outputs):
            instruction = parse_generation_output(raw_output)
            if instruction is None:
                print(f"  [WARN] Failed to parse generation output for: {seed_row['harmful_keyword']}")
                failed += 1
                continue
            generation_records.append(make_record(seed_row, instruction))
            if args.log_path:
                log_records.append({
                    "step": "generation",
                    "seed_harmful_keyword": seed_row["harmful_keyword"],
                    "pseudo_instruction": seed_row["pseudo_instruction"],
                    "raw_output": raw_output,
                })

        append_jsonl(args.save_path, generation_records)
        append_jsonl(args.log_path, log_records)
        total_saved += len(generation_records)
        print(f"  Saved {len(generation_records)} generated records ({failed} failed)")

    print(f"\nDone. Total saved: {total_saved} records → {args.save_path}")


if __name__ == "__main__":
    main()
