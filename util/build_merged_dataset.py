"""Build a merged over-refusal evaluation dataset.

Pseudo-harmful set (label="safe"):
  - ALL k_idioms rows from final_dataset.jsonl, plus
  - final_dataset SAFE rows that gpt-5.5 answered with refusal/deflection, plus
  - OR-Bench / OKTest / PHTest rows that gemini-3.1-flash-lite answered with
    refusal/deflection.
  (Restricting the "refused" pull to benign prompts: for final_dataset only label=="safe"
  rows qualify; the OR-Bench/OKTest/PHTest sets are benign by construction.)

Harmful set (label="unsafe"):
  - The unsafe rows of final_dataset.jsonl, kept as-is.

Unified schema (final_dataset's): prompt, prompt_ko, focus, type, note, label.
  - focus: kept if present, else "".
  - type:  kept if present; OR-Bench uses its `category`; OKTest/PHTest have no
           category, so the source name ("oktest"/"phtest") is used for provenance.
  - note:  kept if present, else "".
"""
import argparse
import json
import os
from typing import Dict, List

FR_ROOT = "/data/home/mk05/FR"

REFUSAL_LABELS = {"refusal", "deflection"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--final_dataset", default=f"{FR_ROOT}/dataset/test/final_dataset.jsonl",
                   help="k_idioms + xstest source (harmful set + k_idioms come from here).")
    p.add_argument("--gpt55_eval", default=f"{FR_ROOT}/outputs/remote_models/gpt-5.5/xstest/evaluation_results.jsonl",
                   help="gpt-5.5 responses on final_dataset, with refusal_clf_label.")
    p.add_argument("--gemini_orbench", default=f"{FR_ROOT}/outputs/remote_models/gemini-3.1-flash-lite/orbench/evaluation_results.jsonl")
    p.add_argument("--gemini_oktest", default=f"{FR_ROOT}/outputs/remote_models/gemini-3.1-flash-lite/oktest/evaluation_results.jsonl")
    p.add_argument("--gemini_phtest", default=f"{FR_ROOT}/outputs/remote_models/gemini-3.1-flash-lite/phtest/evaluation_results.jsonl")
    p.add_argument("--save_path", default=f"{FR_ROOT}/dataset/final/merged_dataset.jsonl")
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record(prompt, prompt_ko, focus, type_, note, label) -> Dict:
    return {
        "prompt": str(prompt or "").strip(),
        "prompt_ko": str(prompt_ko or "").strip(),
        "focus": str(focus or "").strip(),
        "type": str(type_ or "").strip(),
        "note": str(note or "").strip(),
        "label": label,
    }


def is_refused(row: Dict) -> bool:
    return row.get("refusal_clf_label") in REFUSAL_LABELS


def main():
    args = parse_args()

    pseudo: List[Dict] = []

    # --- final_dataset: all k_idioms + safe rows gpt-5.5 refused/deflected ---
    final_eval = read_jsonl(args.gpt55_eval)
    n_kidiom = n_final_refused = 0
    for r in final_eval:
        keep_kidiom = r.get("type") == "k_idioms"
        keep_refused = r.get("label") == "safe" and is_refused(r)
        if not (keep_kidiom or keep_refused):
            continue
        pseudo.append(record(r.get("prompt"), r.get("prompt_ko"), r.get("focus"),
                             r.get("type"), r.get("note"), "safe"))
        n_kidiom += keep_kidiom
        n_final_refused += (keep_refused and not keep_kidiom)

    # --- OR-Bench / OKTest / PHTest: gemini refused/deflected (all benign) ---
    src_counts = {}
    for path, prompt_col, type_fn in [
        (args.gemini_orbench, "prompt", lambda r: r.get("category")),   # category -> type
        (args.gemini_oktest, "prompt", lambda r: "oktest"),             # no category -> source name
        (args.gemini_phtest, "Request", lambda r: "phtest"),           # Request is the English prompt
    ]:
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))  # model dir's benchmark subdir
        bench = os.path.basename(os.path.dirname(path))
        kept = 0
        for r in read_jsonl(path):
            if not is_refused(r):
                continue
            pseudo.append(record(r.get(prompt_col), r.get("prompt_ko"), "", type_fn(r), "", "safe"))
            kept += 1
        src_counts[bench] = kept

    # --- de-duplicate the pseudo set by Korean prompt (keep first occurrence) ---
    seen = set()
    deduped_pseudo = []
    for rec in pseudo:
        key = rec["prompt_ko"]
        if key in seen:
            continue
        seen.add(key)
        deduped_pseudo.append(rec)
    dropped_dups = len(pseudo) - len(deduped_pseudo)

    # --- harmful set: final_dataset unsafe rows, as-is ---
    final_rows = read_jsonl(args.final_dataset)
    harmful = [
        record(r.get("prompt"), r.get("prompt_ko"), r.get("focus"), r.get("type"), r.get("note"), "unsafe")
        for r in final_rows if r.get("label") == "unsafe"
    ]

    merged = deduped_pseudo + harmful

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== pseudo-harmful (label=safe) ===")
    print(f"  final k_idioms (all)         : {n_kidiom}")
    print(f"  final safe & gpt5.5 refused  : {n_final_refused}")
    print(f"  orbench gemini refused       : {src_counts.get('orbench')}")
    print(f"  oktest  gemini refused       : {src_counts.get('oktest')}")
    print(f"  phtest  gemini refused       : {src_counts.get('phtest')}")
    print(f"  duplicates dropped           : {dropped_dups}")
    print(f"  pseudo total                 : {len(deduped_pseudo)}")
    print(f"=== harmful (label=unsafe) from final_dataset: {len(harmful)} ===")
    print(f"TOTAL {len(merged)} -> {args.save_path}")


if __name__ == "__main__":
    main()
