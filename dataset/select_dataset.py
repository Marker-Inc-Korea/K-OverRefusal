import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import GenerationArgs, UniversalGenParams, VLMInferenceEngine
from util import run_batch_generate
try:
    from generation_util import SEED_INSTRUCTION_GENERATION_PROMPT
except ImportError:
    from dataset.generation_util import SEED_INSTRUCTION_GENERATION_PROMPT


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

    parser.add_argument("--seed_word_path", type=str, default=DEFAULT_SEED_WORD_PATH)
    parser.add_argument("--pseudo_keyword_col", type=int, default=0)
    parser.add_argument("--pseudo_instruction_col", type=int, default=1)
    parser.add_argument("--harmful_keyword_col", type=int, default=2)

    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--log_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_num_seeds", type=int, default=None)
    parser.add_argument("--start_seed_index", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max_retries", type=int, default=10)

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




def build_generation_prompt(pseudo_instruction: str, harmful_keyword: str) -> str:
    return SEED_INSTRUCTION_GENERATION_PROMPT.format(
        pseudo_instruction=pseudo_instruction,
        harmful_keyword=harmful_keyword,
    )


_PUNCT = r"[.?!？。！]"

_REFUSAL_PHRASES = (
    "I'm sorry", "I cannot", "I can't", "I am sorry",
    "죄송합니다", "죄송해요", "도움을 드릴 수 없", "도와드릴 수 없",
)


def _last_sentence_in_quotes(text: str) -> Optional[str]:
    for pattern in (rf"`([^`]{{5,}}{_PUNCT})`", rf'"([^"]{{5,}}{_PUNCT})"'):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def _parse_output_tag(text: str) -> Optional[str]:
    # Direct prefix check (handles edge-cases where regex anchor fails)
    if text.startswith("[OUTPUT]:"):
        candidate = text[len("[OUTPUT]:"):].strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate

    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate
    return None


def _strip_output_tag(text: str) -> str:
    if text.startswith("[OUTPUT]:"):
        return text[len("[OUTPUT]:"):].strip()
    return text


def parse_generation_output(text: str) -> Optional[str]:
    text = str(text or "").strip()

    result = _parse_output_tag(text)
    if result:
        return result

    # Fallback: last quoted sentence
    quoted = _last_sentence_in_quotes(text)
    if quoted:
        return quoted

    # Fallback: last non-empty line ending with sentence-final punctuation
    for line in reversed(text.splitlines()):
        line = line.strip()
        if len(line) < 5:
            continue
        if line.startswith("[OUTPUT]:"):
            candidate = line[len("[OUTPUT]:"):].strip()
            if candidate and len(candidate) > 3:
                return candidate
        if line.startswith(("[", "<", "*", "#", "-")):
            continue
        if re.search(r"[.?!？。！]$", line):
            return line

    # Defensive final strip — ensure [OUTPUT]: never leaks into the saved record
    return _strip_output_tag(text) or None


def needs_retry(raw_output: str, parsed: Optional[str]) -> bool:
    """Return True if the generation should be retried."""
    if parsed is None:
        return True
    # Parsed text itself is a refusal (e.g. model put refusal after [OUTPUT]:)
    if any(phrase in parsed for phrase in _REFUSAL_PHRASES):
        return True
    # If there's a clean [OUTPUT]: tag and parsed is not a refusal, trust it
    if re.search(r"^\[OUTPUT\]:\s*.+", raw_output, re.MULTILINE):
        return False
    # No [OUTPUT]: tag — retry if refusal phrases appear anywhere in raw output
    if any(phrase in raw_output for phrase in _REFUSAL_PHRASES):
        return True
    return False


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

    print(f"Generating instructions for {len(seed_rows)} seeds...")

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

    # done[i] = (parsed_instruction, raw_output) for seed_rows[i]
    done: Dict[int, tuple] = {}
    retry_indices = list(range(len(seed_rows)))

    for attempt in range(args.max_retries + 1):
        if not retry_indices:
            break

        label = "Generating" if attempt == 0 else f"Retry {attempt}/{args.max_retries}"
        print(f"{label} for {len(retry_indices)} seeds...")

        prompts = [
            build_generation_prompt(seed_rows[i]["pseudo_instruction"], seed_rows[i]["harmful_keyword"])
            for i in retry_indices
        ]
        raw_outputs = run_batch_generate(model, prompts, gen_params, args.batch_size)

        still_failing = []
        for idx, raw_output in zip(retry_indices, raw_outputs):
            parsed = parse_generation_output(raw_output)
            if needs_retry(raw_output, parsed):
                still_failing.append(idx)
            else:
                done[idx] = (parsed, raw_output)

        if still_failing:
            if attempt < args.max_retries:
                print(f"  → {len(still_failing)} seeds will be retried (refusal or parse failure)")
            else:
                print(f"  → {len(still_failing)} seeds failed after all retries")
                for i in still_failing:
                    print(f"    [FAIL] {seed_rows[i]['harmful_keyword']}")

        retry_indices = still_failing

    records = []
    log_records = []
    for i in sorted(done):
        seed_row = seed_rows[i]
        instruction, raw_output = done[i]
        records.append(make_record(seed_row, instruction))
        if args.log_path:
            log_records.append({
                "seed_harmful_keyword": seed_row["harmful_keyword"],
                "pseudo_instruction": seed_row["pseudo_instruction"],
                "raw_output": raw_output,
                "parsed_instruction": instruction,
            })

    append_jsonl(args.save_path, records)
    append_jsonl(args.log_path, log_records)

    failed = len(retry_indices)
    print(f"\nDone. Saved {len(records)} records ({failed} failed) → {args.save_path}")


if __name__ == "__main__":
    main()
