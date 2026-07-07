"""Assemble the K-OverRefusal eval set with the example-similarity-matched harmful contrast.

NON-DESTRUCTIVE: reads dataset/final/merged_dataset_categorized.jsonl (kept as-is) and writes a
new dataset/final/merged_dataset_v2.jsonl.

Kept untouched (per user):
  - k_idioms (safe, 50) and contrast_k_idioms (unsafe, 50)   [the k_idioms pair]
  - general benign / pseudo-harmful (safe, 135)

Replaced:
  - the old XSTest-derived general contrast (unsafe, 200)  ->  the harmful set produced by
    dataset/embed_match.py, where each harmful prompt is the nearest (open-source e5-large
    embedding) genuinely-harmful neighbour of a general benign prompt. Sources: HarmBench /
    StrongREJECT / SALAD-Bench (KJB Korean) + MaliciousInstruct (open-source instructTrans
    re-translation). Decontaminated vs the K-SafeGuard train set. NO API anywhere.

Eval rows use the ORIGINAL 7-key schema (prompt, prompt_ko, focus, type, note, label, orig_type);
domain is encoded in `type` (harm_<domain>), source in `orig_type`. The benign<->harmful pairing
and similarity live in the provenance sidecar.
"""
import argparse, json, re, unicodedata
from collections import Counter
from pathlib import Path

FR = Path("/data/home/mk05/FR")
CATEG = FR / "dataset/final/merged_dataset_categorized.jsonl"
MATCHED = FR / "dataset/harmful_pool/harmful_matched.jsonl"   # from embed_match.py
KS = Path("/data1/mk05/projects/KS/dataset/final")
OUT = FR / "dataset/final/merged_dataset_v2.jsonl"
PROV = FR / "dataset/final/merged_dataset_v2_harmful_provenance.jsonl"
STATS = FR / "dataset/final/merged_dataset_v2_stats.json"

BENIGN_KEEP = {"k_idioms", "legality_framing", "privacy", "safe_contexts",
               "euphemism", "harmless_analog", "benign_purpose", "other"}
KIDIOM_UNSAFE_KEEP = {"contrast_k_idioms"}

def norm(s):
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", str(s or "")).lower()).strip()

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_harmful", type=int, default=250)
    args = ap.parse_args()

    rows = read(CATEG)
    safe_keep = [r for r in rows if r.get("label") == "safe" and r.get("type") in BENIGN_KEEP]
    kidiom_unsafe = [r for r in rows if r.get("label") == "unsafe" and r.get("type") in KIDIOM_UNSAFE_KEEP]
    kept_norm = {norm(r["prompt_ko"]) for r in safe_keep + kidiom_unsafe}
    print(f"kept safe={len(safe_keep)} (k_idioms+general benign), contrast_k_idioms={len(kidiom_unsafe)}")

    ks_en, ks_ko = set(), set()
    for fn in ["train.jsonl", "train_fr_augmented.jsonl"]:
        fp = KS / fn
        if fp.exists():
            for r in read(fp):
                ks_en.add(norm(r.get("prompt"))); ks_ko.add(norm(r.get("prompt_ko")))

    matched = read(MATCHED)[: args.target_harmful]
    print(f"similarity-matched harmful: {len(matched)} (from {MATCHED.name})")

    new_harmful, prov, seen = [], [], set()
    for r in matched:
        k = norm(r["prompt_ko"])
        if not k or k in seen or k in kept_norm:
            continue
        if k in ks_ko or norm(r.get("prompt_en", "")) in ks_en:
            continue  # KS contamination guard
        seen.add(k)
        dom = r.get("domain", "other")
        new_harmful.append({
            "prompt": r["prompt_ko"], "prompt_ko": r["prompt_ko"], "focus": "",
            "type": f"harm_{dom}", "note": "", "label": "unsafe",
            "orig_type": r.get("source", ""),
        })
        prov.append({"prompt_ko": r["prompt_ko"], "prompt_en": r.get("prompt_en", ""),
                     "source": r.get("source", ""), "domain": dom, "orig_cat": r.get("orig_cat", ""),
                     "matched_benign": r.get("matched_benign", ""),
                     "matched_benign_type": r.get("matched_benign_type", ""),
                     "sim": r.get("sim", None)})

    out_rows = safe_keep + kidiom_unsafe + new_harmful
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(PROV, "w", encoding="utf-8") as f:
        for r in prov:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # verification + stats
    all_norm = [norm(r["prompt_ko"]) for r in out_rows]
    dup = len(all_norm) - len(set(all_norm))
    ks_hits = sum(1 for p in prov if norm(p["prompt_ko"]) in ks_ko or norm(p["prompt_en"]) in ks_en)
    sims = [p["sim"] for p in prov if isinstance(p.get("sim"), (int, float))]
    stats = {
        "total": len(out_rows),
        "safe": sum(1 for r in out_rows if r["label"] == "safe"),
        "unsafe": sum(1 for r in out_rows if r["label"] == "unsafe"),
        "new_harmful": len(new_harmful),
        "new_harmful_by_domain": dict(Counter(p["domain"] for p in prov)),
        "new_harmful_by_source": dict(Counter(p["source"] for p in prov)),
        "matched_benign_mechanism": dict(Counter(p["matched_benign_type"] for p in prov)),
        "similarity_mean": round(sum(sims) / len(sims), 4) if sims else None,
        "similarity_min": min(sims) if sims else None,
        "internal_dup_prompt_ko": dup,
        "new_harmful_KS_train_overlap": ks_hits,
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n=== STATS ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    assert dup == 0, f"internal duplicate prompt_ko: {dup}"
    assert ks_hits == 0, f"KS train overlap: {ks_hits}"
    print(f"\nwrote {OUT}  (dup=0, KS-overlap=0 verified)")

if __name__ == "__main__":
    main()
