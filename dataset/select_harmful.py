"""Select the general harmful contrast candidates for K-OverRefusal (translation+filter path).

Source: dataset/harmful_pool/core_pool.jsonl — already Korean-translated (from the K-JailBreak
pool: HarmBench / StrongREJECT / SALAD-Bench), with AdvBench & JailbreakBench excluded and every
row that collides with the K-SafeGuard train set removed.

Here we only SELECT (no LLM rewriting): a domain-balanced, multi-source subset whose harm domains
correspond to the topical footprint of the benign general set. Human-curated sources
(HarmBench/StrongREJECT) are preferred over SALAD within each domain. Output feeds the
gpt-oss-120b (vLLM) translation-quality filter.

Out: dataset/harmful_pool/harmful_candidates.jsonl  (prompt, prompt_ko, source, domain, orig_cat)
"""
import json, random, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FR = Path("/data/home/mk05/FR")
POOL = FR / "dataset/harmful_pool/core_pool.jsonl"
BENIGN = FR / "dataset/final/merged_dataset_categorized.jsonl"
OUT = FR / "dataset/harmful_pool/harmful_candidates.jsonl"
random.seed(20260707)

# oversample vs the ~200 final target so the LLM filter can drop ~15% and still balance.
DOMAIN_TARGET = {
    "illegal_other":   40,   # broad illegal how-to  (~ legality_framing topics)
    "financial":       22,   # money crime           (~ euphemism topics)
    "fraud_forgery":   16,   # deception/forgery     (~ euphemism topics)
    "privacy_surveil": 30,   # doxxing/surveillance  (~ privacy topics)
    "violence":        30,   # violent crime         (~ safe_contexts / k_idioms topics)
    "weapons":         18,   # weapons/explosives    (~ harmless_analog / benign_purpose)
    "chem_bio":        18,   # chem/bio              (~ harmless_analog / benign_purpose)
    "cyber":           22,   # hacking/malware       (~ benign_purpose / harmless_analog)
    "drugs":           16,   # illegal drugs
    "hate_harass":     22,   # hate/harassment       (~ other)
    "misinfo":         14,   # disinformation        (~ other)
    "illegal_goods":   12,   # weapons/drugs trade
}  # sum = 260 candidates -> ~200 after filter

def norm(s):
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", str(s or "")).lower()).strip()

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

pool = read(POOL)
benign_norm = {norm(r["prompt_ko"]) for r in read(BENIGN)}

by_dom = defaultdict(list)
for r in pool:
    by_dom[r["domain"]].append(r)

# within a domain: human-curated first, shuffled inside each source tier for variety
for d, rows in by_dom.items():
    hc = [r for r in rows if r["source"] in ("harmbench", "strongreject")]
    sl = [r for r in rows if r["source"] not in ("harmbench", "strongreject")]
    random.shuffle(hc); random.shuffle(sl)
    by_dom[d] = hc + sl

selected, seen = [], set()
short_domains = {}
for dom, want in DOMAIN_TARGET.items():
    got = 0
    for r in by_dom.get(dom, []):
        k = norm(r["prompt_ko"])
        if not k or k in seen or k in benign_norm:
            continue
        # keep prompts a reasonable length (avoid multi-paragraph contextual behaviors)
        if len(r["prompt_ko"]) > 160:
            continue
        seen.add(k)
        selected.append({"prompt": r["prompt_en"], "prompt_ko": r["prompt_ko"],
                         "source": r["source"], "domain": dom, "orig_cat": r["orig_cat"]})
        got += 1
        if got >= want:
            break
    if got < want:
        short_domains[dom] = (got, want)

random.shuffle(selected)
with open(OUT, "w", encoding="utf-8") as f:
    for r in selected:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"selected {len(selected)} -> {OUT}")
print("by domain:", dict(Counter(r["domain"] for r in selected)))
print("by source:", dict(Counter(r["source"] for r in selected)))
if short_domains:
    print("SHORT (got,want):", short_domains)
# sanity: none collide with benign
assert not ({norm(r["prompt_ko"]) for r in selected} & benign_norm), "benign collision!"
print("benign-collision check: OK")
