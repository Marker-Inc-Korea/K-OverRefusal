"""Example-level benign<->harmful similarity matching (open-source embedder, local; NO API).

For each general benign (pseudo-harmful) prompt, pick the most semantically/structurally similar
genuinely-harmful prompt from the decontaminated clean pool, so the harmful contrast set mirrors
the benign set example-by-example (not just by domain). Round-robin over benign examples selects
each benign's nearest-unused harmful until the target size is reached, giving fair coverage while
keeping every harmful a near-neighbor of some benign.

Embedder: intfloat/multilingual-e5-large (local transformers, CPU). Pool is rebuilt API-free:
  batch1 (harmful_filtered) + batch2 minus MI + open-source MaliciousInstruct (maliciousinstruct_ko).

Out: dataset/harmful_pool/harmful_matched.jsonl
  {prompt_ko, prompt_en, source, domain, matched_benign, matched_benign_type, sim}
"""
import argparse, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FR = Path("/data/home/mk05/FR")
BENIGN = FR / "dataset/final/merged_dataset_categorized.jsonl"
KS = Path("/data1/mk05/projects/KS/dataset/final")
F1 = FR / "dataset/harmful_pool/harmful_filtered.jsonl"       # batch1 (no MI)
F2 = FR / "dataset/harmful_pool/harmful_filtered_2.jsonl"     # batch2 (drop its API-MI rows)
MI = FR / "dataset/harmful_pool/maliciousinstruct_ko.jsonl"   # open-source re-translation (+filter)
OUT = FR / "dataset/harmful_pool/harmful_matched.jsonl"
CLEAN_POOL = FR / "dataset/harmful_pool/harmful_clean_pool.jsonl"
MODEL_ID = "intfloat/multilingual-e5-large"
GENERAL_BENIGN = {"legality_framing", "privacy", "safe_contexts", "euphemism",
                  "harmless_analog", "benign_purpose", "other"}  # exclude k_idioms (has its own contrast)

def norm(s):
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", str(s or "")).lower()).strip()

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def mi_domain(en):
    c = (en or "").lower()
    if any(k in c for k in ["hack","malware","ddos","phish","exploit","password","keylog","botnet","ransomware"]): return "cyber"
    if any(k in c for k in ["bomb","explos","weapon","gun","firearm","grenade","napalm"]): return "weapons"
    if any(k in c for k in ["drug","meth","cocaine","heroin","fentanyl","narcotic"]): return "drugs"
    if any(k in c for k in ["poison","toxin","chemical","biolog","virus","pathogen","nerve agent"]): return "chem_bio"
    if any(k in c for k in ["launder","counterfeit","fraud","embezzle","tax","insider","scam","forge","fake id","identity theft"]): return "financial"
    if any(k in c for k in ["stalk","track","surveil","spy","private information","personal information","dox"]): return "privacy_surveil"
    if any(k in c for k in ["kill","murder","assault","attack someone","hurt","torture","stab"]): return "violence"
    if any(k in c for k in ["racist","hate","discriminat","harass","bully","manipulat","blackmail","threaten"]): return "hate_harass"
    if any(k in c for k in ["misinformation","disinformation","propaganda","fake news","rumor"]): return "misinfo"
    return "illegal_other"

def build_pool(benign_norm, ks_en, ks_ko):
    pool, seen = [], set()
    def add(prompt_en, prompt_ko, source, domain, orig_cat=""):
        k = norm(prompt_ko)
        if not k or k in seen or k in benign_norm or k in ks_ko or norm(prompt_en) in ks_en:
            return
        seen.add(k)
        pool.append({"prompt_ko": prompt_ko.strip(), "prompt_en": (prompt_en or "").strip(),
                     "source": source, "domain": domain, "orig_cat": orig_cat})
    for r in read(F1):
        if r.get("translation_ok", True):
            add(r.get("prompt"), r.get("prompt_ko"), r.get("source"), r.get("domain", "illegal_other"), r.get("orig_cat", ""))
    for r in read(F2):
        if r.get("translation_ok", True) and r.get("source") != "maliciousinstruct":
            add(r.get("prompt"), r.get("prompt_ko"), r.get("source"), r.get("domain", "illegal_other"), r.get("orig_cat", ""))
    if MI.exists():
        for r in read(MI):
            if r.get("translation_ok", True):
                add(r.get("prompt"), r.get("prompt_ko"), "maliciousinstruct", mi_domain(r.get("prompt")), "")
    else:
        print(f"WARN: {MI} missing — open-source MaliciousInstruct not yet available.")
    return pool

def embed(texts, batch=64):
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    out = []
    for i in range(0, len(texts), batch):
        chunk = ["query: " + t for t in texts[i:i+batch]]
        b = tok(chunk, padding=True, truncation=True, max_length=192, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**b)
        h = o.last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1)
        out.append(F.normalize(emb, p=2, dim=1).cpu())
    import torch as _t
    return _t.cat(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--cap", type=int, default=3, help="max harmful matched per benign example")
    ap.add_argument("--min_sim", type=float, default=0.0, help="drop matches below this cosine")
    args = ap.parse_args()
    import random; random.seed(0)

    def degen(p):  # skip a few broken benign rows (translation artifacts) as MATCH ANCHORS only
        p = (p or "").strip()
        if len(p) < 12:
            return True
        low = p.lower()
        if any(s in low for s in ("followed only", "no explanation", "no markdown", "translation")):
            return True
        return p in ("무엇입니까?", "어떻게 ~하나요?", "어떻게 하나요?")
    benign = [r for r in read(BENIGN)
              if r.get("label") == "safe" and r.get("type") in GENERAL_BENIGN and not degen(r["prompt_ko"])]
    benign_all_norm = {norm(r["prompt_ko"]) for r in read(BENIGN)}
    ks_en, ks_ko = set(), set()
    for fn in ["train.jsonl", "train_fr_augmented.jsonl"]:
        fp = KS / fn
        if fp.exists():
            for r in read(fp):
                ks_en.add(norm(r.get("prompt"))); ks_ko.add(norm(r.get("prompt_ko")))

    pool = build_pool(benign_all_norm, ks_en, ks_ko)
    print(f"benign general={len(benign)}  clean pool={len(pool)}")
    # persist rebuilt API-free clean pool
    with open(CLEAN_POOL, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    be = embed([r["prompt_ko"] for r in benign])
    pe = embed([r["prompt_ko"] for r in pool])
    sims = (be @ pe.T)  # [B, P]

    import torch
    order_per_benign = [torch.argsort(sims[i], descending=True).tolist() for i in range(len(benign))]
    cursor = [0] * len(benign)
    used = set()
    per_benign_cnt = Counter()
    selected = []
    rounds = 0
    while len(selected) < args.target and rounds < args.cap:
        idxs = list(range(len(benign))); random.shuffle(idxs)
        progressed = False
        for bi in idxs:
            if len(selected) >= args.target:
                break
            if per_benign_cnt[bi] >= args.cap:
                continue
            # advance to next unused pool item for this benign
            while cursor[bi] < len(pool) and order_per_benign[bi][cursor[bi]] in used:
                cursor[bi] += 1
            if cursor[bi] >= len(pool):
                continue
            pj = order_per_benign[bi][cursor[bi]]
            sim = float(sims[bi][pj])
            if sim < args.min_sim:
                continue
            used.add(pj); per_benign_cnt[bi] += 1; cursor[bi] += 1; progressed = True
            r = pool[pj]
            selected.append({**r, "matched_benign": benign[bi]["prompt_ko"],
                             "matched_benign_type": benign[bi]["type"], "sim": round(sim, 4)})
        rounds += 1
        if not progressed:
            break

    with open(OUT, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sims_list = [r["sim"] for r in selected]
    print(f"\nselected {len(selected)} -> {OUT}")
    print("by source:", dict(Counter(r["source"] for r in selected)))
    print("by domain:", dict(Counter(r["domain"] for r in selected)))
    print("by matched benign mechanism:", dict(Counter(r["matched_benign_type"] for r in selected)))
    if sims_list:
        import statistics as st
        print(f"sim: min={min(sims_list):.3f} mean={st.mean(sims_list):.3f} max={max(sims_list):.3f} "
              f"p10={sorted(sims_list)[len(sims_list)//10]:.3f}")
    print("\n--- lowest-similarity matches (inspect) ---")
    for r in sorted(selected, key=lambda x: x["sim"])[:6]:
        print(f"  sim={r['sim']:.3f} [{r['domain']}] H: {r['prompt_ko'][:48]}  <=>  B: {r['matched_benign'][:40]}")
    print("--- highest-similarity matches ---")
    for r in sorted(selected, key=lambda x: -x["sim"])[:6]:
        print(f"  sim={r['sim']:.3f} [{r['domain']}] H: {r['prompt_ko'][:48]}  <=>  B: {r['matched_benign'][:40]}")

if __name__ == "__main__":
    main()
