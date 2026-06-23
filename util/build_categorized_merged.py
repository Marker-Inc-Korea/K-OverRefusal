"""Reconcile mechanism labels (Opus cross-check vs gpt-oss-120b), then write a
mechanism-categorized + count-balanced version of the pseudo-harmful set.

Inputs:
  - dataset/final/merged_dataset.jsonl        (source; `type` = provenance)
  - dataset/final/mechanism_gptoss.jsonl       (gpt-oss-120b classifier output)
  - util/mechanism_labels_opus.json            (direct Opus labels, cross-check)

Pipeline:
  1. Normalize xstest-native types onto the final mechanism axis (privacy_*->privacy,
     *_discr->discrimination) and keep k_idioms whole.
  2. For orbench/oktest/phtest rows, take the final mechanism from `--primary`
     (opus|gptoss); report Opus vs gpt-oss agreement and list disagreements.
  3. Balance: keep k_idioms whole; for every other mechanism category, drop those
     below `--floor`, and downsample the rest to `--cap` (seeded, deterministic).
  4. Write the balanced pseudo-harmful set + the untouched harmful rows.

Run `--report_only` first to see the distribution before committing to cap/floor.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

FR_ROOT = Path(__file__).resolve().parents[1]

XSTEST_TO_FINAL = {
    "homonyms": "homonyms",
    "figurative_language": "figurative_language",
    "safe_targets": "safe_targets",
    "safe_contexts": "safe_contexts",
    "definitions": "definitions",
    "nons_group_real_discr": "discrimination",
    "real_group_nons_discr": "discrimination",
    "historical_events": "historical_events",
    "privacy_public": "privacy",
    "privacy_fictional": "privacy",
}


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_path", default=f"{FR_ROOT}/dataset/final/merged_dataset.jsonl")
    p.add_argument("--gptoss_path", default=f"{FR_ROOT}/dataset/final/mechanism_gptoss.jsonl")
    p.add_argument("--opus_path", default=f"{FR_ROOT}/util/mechanism_labels_opus.json")
    p.add_argument("--save_path", default=f"{FR_ROOT}/dataset/final/merged_dataset_categorized.jsonl")
    p.add_argument("--primary", choices=["opus", "gptoss"], default="opus",
                   help="Which labeler is authoritative for orbench/oktest/phtest rows.")
    p.add_argument("--cap", type=int, default=18, help="Max rows per non-kidioms mechanism.")
    p.add_argument("--floor", type=int, default=8, help="Drop mechanisms with fewer rows than this.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.merged_path)
    opus = json.load(open(args.opus_path, encoding="utf-8"))["labels"]
    gptoss = {r["idx"]: r["mechanism"] for r in read_jsonl(args.gptoss_path)} if os.path.exists(args.gptoss_path) else {}

    # --- assign final mechanism per safe row ---
    final_mech: Dict[int, str] = {}
    agree = disagree = 0
    disagreements = []
    calib_rows = []  # (idx, xstest_type, gptoss_label) for the 13 xstest-native rows
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
                calib_rows.append((i, final_mech[i], gptoss[i]))
            continue
        o = opus.get(str(i))
        g = gptoss.get(i)
        if o and g:
            if o == g:
                agree += 1
            else:
                disagree += 1
                disagreements.append((i, t, o, g, r.get("prompt", "")[:80]))
        final_mech[i] = (g if args.primary == "gptoss" else o) or g or o or "OTHER"

    # --- reports ---
    raw_dist = Counter(final_mech.values())
    print("=== mechanism distribution (pre-balance) ===")
    for k, v in raw_dist.most_common():
        print(f"  {v:>3}  {k}")
    if gptoss:
        n = agree + disagree
        print(f"\n=== Opus vs gpt-oss agreement (orbench/oktest/phtest, n={n}) ===")
        print(f"  agree {agree}  disagree {disagree}  ->  {100*agree/max(n,1):.1f}%")
        if calib_rows:
            ok = sum(x == g for _, x, g in calib_rows)
            print(f"  gpt-oss vs XSTest-native labels (n={len(calib_rows)}): {ok} match")
        print("  --- disagreements (idx, orig, opus, gptoss, prompt) ---")
        for idx, t, o, g, pr in disagreements:
            print(f"   {idx:>3} [{t}] opus={o} gptoss={g} | {pr}")
    else:
        print("\n[note] gpt-oss output not found yet — reporting with Opus labels only.")

    if args.report_only:
        return

    # --- balance ---
    rng = random.Random(args.seed)
    by_mech = defaultdict(list)
    for i in final_mech:
        by_mech[final_mech[i]].append(i)

    keep_idx = set()
    print("\n=== balanced selection ===")
    for mech, idxs in sorted(by_mech.items(), key=lambda kv: -len(kv[1])):
        if mech == "k_idioms":
            keep_idx.update(idxs)
            print(f"  {len(idxs):>3}  {mech} (kept whole)")
            continue
        if len(idxs) < args.floor:
            print(f"  ---  {mech} dropped ({len(idxs)} < floor {args.floor})")
            continue
        chosen = sorted(rng.sample(idxs, min(args.cap, len(idxs))))
        keep_idx.update(chosen)
        print(f"  {len(chosen):>3}  {mech} (from {len(idxs)})")

    # --- write: balanced pseudo-harmful + untouched harmful ---
    out = []
    for i, r in enumerate(rows):
        if r.get("label") == "safe":
            if i not in keep_idx:
                continue
            rec = dict(r)
            rec["orig_type"] = r.get("type")
            rec["type"] = final_mech[i]
            out.append(rec)
        else:
            rec = dict(r)
            rec.setdefault("orig_type", r.get("type"))
            out.append(rec)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    safe_n = sum(1 for r in out if r["label"] == "safe")
    print(f"\nWrote {len(out)} rows ({safe_n} pseudo-harmful + {len(out)-safe_n} harmful) -> {args.save_path}")


if __name__ == "__main__":
    main()
