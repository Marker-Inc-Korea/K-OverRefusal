"""Batch-2 harmful candidate selection: enlarge the pool + fold in MaliciousInstruct.

Selects an additional domain-balanced set from the (enlarged) decontaminated core pool,
EXCLUDING everything already chosen for batch-1 (harmful_candidates.jsonl), and appends the
freshly-translated MaliciousInstruct rows. Output goes to the gpt-oss-120b(vLLM) filter.

Out: dataset/harmful_pool/harmful_candidates_2.jsonl
"""
import json, random, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FR = Path("/data/home/mk05/FR")
POOL = FR / "dataset/harmful_pool/core_pool.jsonl"
BATCH1 = FR / "dataset/harmful_pool/harmful_candidates.jsonl"
MI = FR / "dataset/harmful_pool/maliciousinstruct_ko.jsonl"
BENIGN = FR / "dataset/final/merged_dataset_categorized.jsonl"
KS = Path("/data1/mk05/projects/KS/dataset/final")
OUT = FR / "dataset/harmful_pool/harmful_candidates_2.jsonl"
random.seed(11)

BATCH2_TARGET = {
    "illegal_other": 40, "financial": 22, "fraud_forgery": 16, "privacy_surveil": 28,
    "violence": 30, "weapons": 18, "chem_bio": 18, "cyber": 22, "drugs": 16,
    "hate_harass": 22, "misinfo": 14, "illegal_goods": 10,
}

def norm(s):
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", str(s or "")).lower()).strip()

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

pool = read(POOL)
benign_norm = {norm(r["prompt_ko"]) for r in read(BENIGN)}
batch1_norm = {norm(r["prompt_ko"]) for r in read(BATCH1)}

ks_en, ks_ko = set(), set()
for fn in ["train.jsonl", "train_fr_augmented.jsonl"]:
    fp = KS / fn
    if fp.exists():
        for r in read(fp):
            ks_en.add(norm(r.get("prompt"))); ks_ko.add(norm(r.get("prompt_ko")))

excluded = benign_norm | batch1_norm
def bad(en, ko):
    k = norm(ko)
    return (not k) or k in excluded or k in ks_ko or norm(en) in ks_en or len(ko) > 160

by_dom = defaultdict(list)
for r in pool:
    by_dom[r["domain"]].append(r)
for d, rows in by_dom.items():
    hc = [r for r in rows if r["source"] in ("harmbench", "strongreject")]
    sl = [r for r in rows if r["source"] not in ("harmbench", "strongreject")]
    random.shuffle(hc); random.shuffle(sl)
    by_dom[d] = hc + sl

selected, seen = [], set()
for dom, want in BATCH2_TARGET.items():
    got = 0
    for r in by_dom.get(dom, []):
        if bad(r["prompt_en"], r["prompt_ko"]):
            continue
        k = norm(r["prompt_ko"])
        if k in seen:
            continue
        seen.add(k)
        selected.append({"prompt": r["prompt_en"], "prompt_ko": r["prompt_ko"],
                         "source": r["source"], "domain": dom, "orig_cat": r["orig_cat"]})
        got += 1
        if got >= want:
            break

# append MaliciousInstruct (dedup vs everything)
mi_added = 0
for r in read(MI):
    if bad(r["prompt"], r["prompt_ko"]):
        continue
    k = norm(r["prompt_ko"])
    if k in seen:
        continue
    seen.add(k)
    selected.append({"prompt": r["prompt"], "prompt_ko": r["prompt_ko"],
                     "source": "maliciousinstruct", "domain": r["domain"], "orig_cat": ""})
    mi_added += 1

random.shuffle(selected)
with open(OUT, "w", encoding="utf-8") as f:
    for r in selected:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"batch2 selected {len(selected)} (incl MaliciousInstruct {mi_added}) -> {OUT}")
print("by domain:", dict(Counter(r["domain"] for r in selected)))
print("by source:", dict(Counter(r["source"] for r in selected)))
assert not ({norm(r["prompt_ko"]) for r in selected} & (excluded)), "overlap with benign/batch1!"
print("dedup vs benign+batch1: OK")
