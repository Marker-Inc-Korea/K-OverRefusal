"""Build a decontaminated, domain-tagged harmful-core pool for the mechanism-mirrored
harmful contrast set.

Sources (all Korean-translated, from the K-JailBreak pool at /data/home/mk05/KJB/dataset/ko):
  - HarmBench      (human-curated red-team behaviors; SemanticCategory labels)
  - StrongREJECT   (curated forbidden prompts; category labels)
  - SALAD-Bench    (base behaviors, method=None only -> no jailbreak wrapper; 3-level taxonomy)

Excluded entirely (present in the K-SafeGuard moderator train set -> would contaminate eval):
  - AdvBench, JailbreakBench.
Additionally every candidate is dropped if its normalized English OR Korean prompt collides
with the K-SafeGuard train set (train.jsonl + train_fr_augmented.jsonl).

Output: dataset/harmful_pool/core_pool.jsonl
  schema: {prompt_ko, prompt_en, source, domain, orig_cat}
"""
import json, re, unicodedata, random
from collections import Counter, defaultdict
from pathlib import Path

KJB = Path("/data/home/mk05/KJB/dataset/ko")
KS = Path("/data1/mk05/projects/KS/dataset/final")
OUT = Path("/data/home/mk05/FR/dataset/harmful_pool/core_pool.jsonl")
random.seed(0)

def norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"[^\w가-힣]", "", s).strip()

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---- KS train exclusion set ----
ks_en, ks_ko = set(), set()
for fn in ["train.jsonl", "train_fr_augmented.jsonl"]:
    fp = KS / fn
    if fp.exists():
        for r in read(fp):
            if r.get("prompt"): ks_en.add(norm(r["prompt"]))
            if r.get("prompt_ko"): ks_ko.add(norm(r["prompt_ko"]))
print(f"KS exclusion: EN={len(ks_en)} KO={len(ks_ko)}")

def contaminated(en, ko):
    return norm(en) in ks_en or norm(ko) in ks_ko

# ---- coarse domain from source-specific labels ----
def salad_domain(c3):
    c = (c3 or "").lower()
    if any(k in c for k in ["hate", "stereotyp", "insult", "harass", "threaten", "bully", "discrimin"]): return "hate_harass"
    if "privacy" in c or "surveillance" in c or "doxx" in c: return "privacy_surveil"
    if "forge identity" in c or "identity" in c: return "fraud_forgery"
    if any(k in c for k in ["fraud", "financial", "scam", "counterfeit", "money"]): return "financial"
    if "plagiar" in c or "academic" in c or "cheat" in c: return "fraud_forgery"
    if "theft" in c or "robbery" in c: return "financial"
    if any(k in c for k in ["disinformation", "misinformation", "influence", "erosion", "misconception", "defamation", "rumor"]): return "misinfo"
    if "drug" in c or "substance" in c: return "drugs"
    if "weapon" in c or "firearm" in c or "explos" in c: return "weapons"
    if any(k in c for k in ["violent", "violence", "pain", "murder", "assault", "kidnap"]): return "violence"
    if "self-harm" in c or "suicide" in c: return "self_harm"
    if any(k in c for k in ["erotic", "porn", "sexual", "adult"]): return "sexual"
    if any(k in c for k in ["hack", "malware", "cyber", "security", "phishing"]): return "cyber"
    if "dangerous" in c or "cbrn" in c or "biolog" in c or "chemical" in c: return "chem_bio"
    return "illegal_other"

HB_DOMAIN = {
    "chemical_biological": "chem_bio",
    "cybercrime_intrusion": "cyber",
    "illegal": "illegal_other",
    "misinformation_disinformation": "misinfo",
    "harassment_bullying": "hate_harass",
    "harmful": "violence",
}
SR_DOMAIN = {
    "Non-violent crimes": "illegal_other",
    "Violence": "violence",
    "Disinformation and deception": "misinfo",
    "Hate, harassment and discrimination": "hate_harass",
    "Illegal goods and services": "illegal_goods",
    "Sexual content": "sexual",
}

pool = []
seen_ko = set()

def add(prompt_en, prompt_ko, source, domain, orig_cat):
    k = norm(prompt_ko)
    if not k or k in seen_ko: return False
    if contaminated(prompt_en, prompt_ko): return False
    seen_ko.add(k)
    pool.append({"prompt_ko": prompt_ko.strip(), "prompt_en": (prompt_en or "").strip(),
                 "source": source, "domain": domain, "orig_cat": orig_cat})
    return True

# HarmBench: skip copyright (FunctionalCategory) — IP, not a danger domain we mirror
for r in read(KJB / "harmbench.jsonl"):
    if not r.get("translation_ok", True): continue
    if r.get("FunctionalCategory") == "copyright" or r.get("SemanticCategory") == "copyright": continue
    dom = HB_DOMAIN.get(r.get("SemanticCategory"), "illegal_other")
    add(r.get("prompt"), r.get("prompt_ko"), "harmbench", dom, r.get("SemanticCategory"))

# StrongREJECT
for r in read(KJB / "strongreject.jsonl"):
    if not r.get("translation_ok", True): continue
    dom = SR_DOMAIN.get(r.get("category"), "illegal_other")
    add(r.get("prompt"), r.get("prompt_ko"), "strongreject", dom, r.get("category"))

# SALAD base (method=None only), dedup + cap per 3-category for diversity
salad_none = [r for r in read(KJB / "salad_bench.jsonl")
              if r.get("method") is None and r.get("translation_ok", True)]
random.shuffle(salad_none)
cap = Counter()
for r in salad_none:
    c3 = r.get("3-category")
    dom = salad_domain(c3)
    if cap[c3] >= 300:  # keep pool broad but bounded (raised to enlarge the pool)
        continue
    if add(r.get("prompt"), r.get("prompt_ko"), "salad_bench", dom, c3):
        cap[c3] += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in pool:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nPool size: {len(pool)}  -> {OUT}")
print("by source:", dict(Counter(r["source"] for r in pool)))
print("\nby domain:")
for d, n in Counter(r["domain"] for r in pool).most_common():
    src = Counter(r["source"] for r in pool if r["domain"] == d)
    print(f"  {d:16s} {n:5d}   {dict(src)}")
