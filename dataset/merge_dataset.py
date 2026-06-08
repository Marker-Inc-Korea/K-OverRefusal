"""
Merge three Korean safety datasets into a single unified JSONL file.

Sources:
  1. test_dataset.jsonl       — manually written pseudo-harmful (keyword-based)
  2. final_match_dataset.jsonl — LLM-generated harmful (keyword-based)
  3. xstest_ko_qwen.jsonl     — translated XSTest (safe + unsafe)

Unified schema (XSTest-based):
  prompt      — original text (English for XSTest; Korean for sources 1&2)
  prompt_ko   — Korean text
  focus       — keyword / topic
  type        — XSTest type string
  note        — free-text note (empty string when not available)
  label       — "safe" | "unsafe"
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


DEFAULT_TEST_PATH = "/data/home/mk05/FR/dataset/test/test_dataset.jsonl"
DEFAULT_MATCH_PATH = "/data/home/mk05/FR/dataset/test/final_match_dataset.jsonl"
DEFAULT_XSTEST_PATH = "/data/home/mk05/FR/dataset/xstest/xstest_ko_qwen.jsonl"
DEFAULT_SAVE_PATH = "/data/home/mk05/FR/dataset/test/final_dataset.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--match_path", type=str, default=DEFAULT_MATCH_PATH)
    parser.add_argument("--xstest_path", type=str, default=DEFAULT_XSTEST_PATH)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _is_broken_prompt_ko(value: str) -> bool:
    return not value or value.strip() in {"`.", "`.\"`}", "`.\"}"}


def convert_keyword_record(record: Dict, record_type: str, label: str) -> Dict:
    instruction = record.get("instruction", "").strip()
    seed = record.get("seed", "")
    return {
        "prompt": instruction,
        "prompt_ko": instruction,
        "focus": seed,
        "type": record_type,
        "note": "",
        "label": label,
    }


def convert_xstest_record(record: Dict) -> Dict:
    return {
        "prompt": record.get("prompt", ""),
        "prompt_ko": record.get("prompt_ko", ""),
        "focus": record.get("focus", ""),
        "type": record.get("type", ""),
        "note": record.get("note", ""),
        "label": record.get("label", ""),
    }


def main():
    args = parse_args()

    results: List[Dict] = []

    # ── Source 1: pseudo-harmful keyword dataset ──────────────────────────────
    test_records = load_jsonl(args.test_path)
    for r in test_records:
        label = r.get("label", "safe")
        results.append(convert_keyword_record(r, "homonyms", label))
    print(f"test_dataset:        {len(test_records):4d} records")

    # ── Source 2: LLM-generated harmful keyword dataset ───────────────────────
    match_records = load_jsonl(args.match_path)
    for r in match_records:
        # final_match_dataset has no label field — all entries are harmful
        results.append(convert_keyword_record(r, "contrast_homonyms", "unsafe"))
    print(f"final_match_dataset: {len(match_records):4d} records")

    # ── Source 3: XSTest (translated) ─────────────────────────────────────────
    xstest_records = load_jsonl(args.xstest_path)
    broken = 0
    for r in xstest_records:
        converted = convert_xstest_record(r)
        if _is_broken_prompt_ko(converted["prompt_ko"]):
            broken += 1
        results.append(converted)
    print(f"xstest_ko:           {len(xstest_records):4d} records  ({broken} broken prompt_ko — re-run translate job)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    safe_count = sum(1 for r in results if r["label"] == "safe")
    unsafe_count = sum(1 for r in results if r["label"] == "unsafe")
    print(f"\nSaved {len(results)} records → {args.save_path}")
    print(f"  safe={safe_count}  unsafe={unsafe_count}")


if __name__ == "__main__":
    main()
