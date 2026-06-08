"""
Re-parse select_dataset.py outputs using corrected [OUTPUT]: extraction.

Root cause of the bug:
  re.search(r'\\[OUTPUT\\]:\\s*(.+)') finds the FIRST [OUTPUT]: in the text,
  which in thinking-mode outputs is often the prompt format description:
    e.g. `[OUTPUT]: <Korean instruction, no explanation>`.
  This causes `<Korean instruction, no explanation>`.` to be saved as the instruction.

Fix:
  Use re.findall with re.MULTILINE to match [OUTPUT]: only at the start of a line,
  then take the LAST match (the model's actual final answer, not a mid-thinking reference).
  Add backtick-quoted sentence as a fallback for truncated outputs where [OUTPUT]:
  was never reached.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PSEUDO_KEYWORD_COL = 0
PSEUDO_INSTRUCTION_COL = 1
HARMFUL_KEYWORD_COL = 2

DEFAULT_LOG_PATH = "/data/home/mk05/FR/dataset/test/qwen36_27b_semantic_match_judge_log.jsonl"
DEFAULT_OUTPUT_PATH = "/data/home/mk05/FR/dataset/test/qwen36_27b_semantic_match_dataset.jsonl"
DEFAULT_SEED_WORD_PATH = "/data/home/mk05/FR/dataset/seed_word.csv"
DEFAULT_SAVE_PATH = "/data/home/mk05/FR/dataset/test/final_match_dataset.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed_word_path", type=str, default=DEFAULT_SEED_WORD_PATH)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()


def read_seed_rows(path: str) -> List[Dict[str, str]]:
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= HARMFUL_KEYWORD_COL:
                continue
            pseudo_keyword = row[PSEUDO_KEYWORD_COL].strip()
            pseudo_instruction = row[PSEUDO_INSTRUCTION_COL].strip()
            harmful_keyword = row[HARMFUL_KEYWORD_COL].strip()
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


def load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def parse_text_from_raw(text: str) -> Optional[str]:
    """
    Extract the final instruction text from a model's raw output.

    Priority:
    1. Last `[OUTPUT]:` that appears at the start of a line — avoids matching
       format descriptions inside the thinking process (e.g. `[OUTPUT]: <placeholder>`).
       Ignores matches that are template placeholders (start with '<').
    2. Last backtick-quoted sentence ending with sentence-final punctuation —
       handles truncated outputs where thinking was cut off before [OUTPUT]:.
    3. Last double-quoted sentence ending with sentence-final punctuation.
    """
    text = str(text or "").strip()
    PUNCT = r"[.?!？。！]"

    # 1. Last start-of-line [OUTPUT]: match
    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate

    # 2. Last backtick-quoted sentence ending with punctuation
    matches = re.findall(rf"`([^`]{{5,}}{PUNCT})`", text)
    if matches:
        return matches[-1].strip()

    # 3. Last double-quoted sentence ending with punctuation
    matches = re.findall(rf'"([^"]{{5,}}{PUNCT})"', text)
    if matches:
        return matches[-1].strip()

    return None


def parse_index_from_raw(text: str, num_candidates: int) -> int:
    """
    Extract selected candidate index from selection output.
    Uses the LAST start-of-line [OUTPUT]: N match.
    """
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


def make_record(seed_pseudo_keyword: str, instruction: str) -> Dict:
    return {
        "output": "",
        "input": "",
        "instruction": instruction,
        "response": "",
        "seed": seed_pseudo_keyword,
    }


def main():
    args = parse_args()

    # ── 1. Load CSV → build key<->seed mappings ──────────────────────────────
    seed_rows = read_seed_rows(args.seed_word_path)
    # (pseudo_instruction, harmful_keyword) → pseudo_keyword
    key_to_seed: Dict[Tuple, str] = {
        (r["pseudo_instruction"], r["harmful_keyword"]): r["pseudo_keyword"]
        for r in seed_rows
    }
    # pseudo_keyword → (pseudo_instruction, harmful_keyword)
    seed_to_key: Dict[str, Tuple] = {v: k for k, v in key_to_seed.items()}
    print(f"Loaded {len(seed_rows)} seed rows from CSV")

    # ── 2. Re-parse log entries ───────────────────────────────────────────────
    log_entries = load_jsonl(args.log_path)
    re_parsed: Dict[Tuple, str] = {}   # key → corrected instruction

    gen_total = gen_ok = 0
    sel_total = sel_ok = 0
    idx_changed = 0

    for entry in log_entries:
        step = entry.get("step", "")
        harmful_keyword = entry.get("seed_harmful_keyword", "")
        pseudo_instruction = entry.get("pseudo_instruction", "")
        raw_output = entry.get("raw_output", "")
        key = (pseudo_instruction, harmful_keyword)

        if step == "generation":
            gen_total += 1
            instruction = parse_text_from_raw(raw_output)
            if instruction:
                re_parsed[key] = instruction
                gen_ok += 1
            else:
                print(f"  [WARN] generation parse failed for: {harmful_keyword!r}")

        elif step == "selection":
            sel_total += 1
            num_candidates = entry.get("num_candidates", 1)
            new_idx = parse_index_from_raw(raw_output, num_candidates)
            old_idx = entry.get("selected_index", -1)
            stored_instruction = entry.get("selected_instruction", "")

            if new_idx != old_idx:
                idx_changed += 1
                print(
                    f"  [INFO] selection idx changed {old_idx}→{new_idx} for: {harmful_keyword!r}"
                    f"  (stored instruction kept — candidates not available)"
                )

            if stored_instruction:
                re_parsed[key] = stored_instruction
                sel_ok += 1
            else:
                print(f"  [WARN] selection has no stored instruction for: {harmful_keyword!r}")

    print(
        f"Log re-parse: generation {gen_ok}/{gen_total} OK | "
        f"selection {sel_ok}/{sel_total} OK ({idx_changed} idx changes)"
    )

    # ── 3. Load main output, replace selection/generation instructions ────────
    main_output = load_jsonl(args.output_path)

    results = []
    re_parsed_count = direct_count = skipped = 0

    for entry in main_output:
        seed_pseudo_keyword = entry.get("seed", "")
        key = seed_to_key.get(seed_pseudo_keyword)

        if key is not None and key in re_parsed:
            instruction = re_parsed[key]
            if not instruction:
                print(f"  [SKIP] empty instruction after re-parse for seed: {seed_pseudo_keyword!r}")
                skipped += 1
                continue
            results.append(make_record(seed_pseudo_keyword, instruction))
            re_parsed_count += 1
        else:
            # Direct entry (single candidate, no LLM call) — already correct
            results.append(entry)
            direct_count += 1

    # ── 4. Save ───────────────────────────────────────────────────────────────
    with open(args.save_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} records → {args.save_path}")
    print(f"  re-parsed: {re_parsed_count} | direct (unchanged): {direct_count} | skipped: {skipped}")

    # ── 5. Quick sanity check ─────────────────────────────────────────────────
    bad = [r for r in results if not r.get("instruction") or r["instruction"].startswith("`") or r["instruction"].startswith("<")]
    if bad:
        print(f"\n[WARN] {len(bad)} records still look suspicious:")
        for r in bad[:5]:
            print(f"  seed={r['seed']!r}  instruction={r['instruction'][:80]!r}")
    else:
        print("\nAll instructions look valid.")


if __name__ == "__main__":
    main()
