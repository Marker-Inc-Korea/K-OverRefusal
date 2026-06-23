"""Reconcile mechanism labels (manual cross-check vs gpt-oss-120b) and write a
mechanism-categorized version of the pseudo-harmful set using ALL rows (no downsampling).

The coarse `safe_framing` mechanism is split (in util/mechanism_labels_manual.json) into
euphemism / legality_framing / benign_purpose, and rare low-frequency mechanisms
(definitions, homonyms, discrimination, ...) are bundled as `other`, so category counts
come out roughly even WITHOUT dropping or downsampling any prompt.

Inputs:
  - dataset/final/merged_dataset.jsonl        (source; `type` = provenance)
  - dataset/final/mechanism_gptoss.jsonl       (gpt-oss-120b classifier output, coarse)
  - util/mechanism_labels_manual.json          (direct manual labels, fine-grained)

Pipeline:
  1. Normalize xstest-native types onto the final axis (privacy_*->privacy, safe_contexts
     kept; rare mechanisms folded into `other`) and keep k_idioms whole.
  2. For orbench/oktest/phtest rows, take the final mechanism from `--primary`
     (manual|gptoss). Report manual vs gpt-oss agreement at the COARSE level (collapsing the
     three safe_framing sub-categories and the `other` members) and list disagreements.
  3. Write every pseudo-harmful row (re-typed) + the untouched harmful rows.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

FR_ROOT = Path(__file__).resolve().parents[1]

# xstest-native provenance type -> final mechanism category.
XSTEST_TO_FINAL = {
    "homonyms": "other",
    "figurative_language": "other",
    "safe_targets": "other",
    "safe_contexts": "safe_contexts",
    "definitions": "other",
    "nons_group_real_discr": "other",
    "real_group_nons_discr": "other",
    "historical_events": "other",
    "privacy_public": "privacy",
    "privacy_fictional": "privacy",
}

# Collapse the final fine-grained labels (and gpt-oss's coarse labels) into a common
# space so the manual labels and gpt-oss can be compared apples-to-apples.
COARSE = {
    "euphemism": "safe_framing", "legality_framing": "safe_framing", "benign_purpose": "safe_framing",
    "definitions": "other", "homonyms": "other", "discrimination": "other",
    "figurative_language": "other", "safe_targets": "other", "historical_events": "other",
    "OTHER": "other",
}


def coarse(x: str) -> str:
    return COARSE.get(x, x)


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", default=f"{FR_ROOT}/dataset/final/merged_dataset.jsonl")
    p.add_argument("--gptoss_path", default=f"{FR_ROOT}/dataset/final/mechanism_gptoss.jsonl")
    p.add_argument("--manual_path", default=f"{FR_ROOT}/util/mechanism_labels_manual.json")
    p.add_argument("--save_path", default=f"{FR_ROOT}/dataset/final/merged_dataset_categorized.jsonl")
    p.add_argument("--primary", choices=["manual", "gptoss"], default="manual",
                   help="Which labeler is authoritative for orbench/oktest/phtest rows.")
    p.add_argument("--report_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.merged_path)
    manual = json.load(open(args.manual_path, encoding="utf-8"))["labels"]
    gptoss = {r["idx"]: r["mechanism"] for r in read_jsonl(args.gptoss_path)} if os.path.exists(args.gptoss_path) else {}

    final_mech: Dict[int, str] = {}
    agree = disagree = 0
    disagreements = []
    calib_rows = []  # (idx, xstest_final, gptoss_coarse) for the xstest-native calibration rows
    for i, r in enumerate(rows):
        if r.get("label") != "safe":
            continue
        t = r.get("type")
        if t == "k_idioms":
            final_mech[i] = "k_idioms"
            continue
        if t in XSTEST_TO_FINAL:
            final_mech[i] = XSTEST_TO_FINAL[t]
            if i in gptoss:
                calib_rows.append((i, final_mech[i], coarse(gptoss[i])))
            continue
        o = manual.get(str(i))
        g = gptoss.get(i)
        if o and g:
            if coarse(o) == coarse(g):
                agree += 1
            else:
                disagree += 1
                disagreements.append((i, t, o, g, r.get("prompt", "")[:80]))
        final_mech[i] = (g if args.primary == "gptoss" else o) or g or o or "other"

    raw_dist = Counter(final_mech.values())
    print("=== mechanism distribution (all rows, no downsampling) ===")
    for k, v in raw_dist.most_common():
        print(f"  {v:>3}  {k}")
    if gptoss:
        n = agree + disagree
        print(f"\n=== manual vs gpt-oss COARSE agreement (orbench/oktest/phtest, n={n}) ===")
        print(f"  agree {agree}  disagree {disagree}  ->  {100*agree/max(n,1):.1f}%")
        if calib_rows:
            ok = sum(x == g for _, x, g in calib_rows)
            print(f"  gpt-oss vs XSTest-native (coarse, n={len(calib_rows)}): {ok} match")

    if args.report_only:
        return

    out = []
    for i, r in enumerate(rows):
        rec = dict(r)
        rec.setdefault("orig_type", r.get("type"))
        if r.get("label") == "safe":
            rec["orig_type"] = r.get("type")
            rec["type"] = final_mech[i]
        out.append(rec)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    safe_n = sum(1 for r in out if r["label"] == "safe")
    print(f"\nWrote {len(out)} rows ({safe_n} pseudo-harmful + {len(out)-safe_n} harmful) -> {args.save_path}")


if __name__ == "__main__":
    main()
