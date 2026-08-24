"""Aggregate per-model K-SafeGuard metrics into reports/<benchmark>/results/model_scores.csv.

    python eval/aggregate_scores.py --benchmark merged_v2 [--evaluator ksafeguard] \
        [--models NAME ...] [--out reports/merged_v2/results/model_scores.csv]

Columns mirror the original merged_v2 table: overall safe_compliance / safe_overrefusal /
unsafe_refusal / behavior_acc / refusal_f1, the k_idioms pair group (kidiom_comp / kidiom_refusal)
and the general group (harm_comp / harm_refusal = benign compliance / harm_* refusal).
Also adds a per-mechanism-category safe-compliance breakdown csv (model_category_scores.csv)
computed from the results jsonl."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="merged_v2")
    ap.add_argument("--evaluator", default="ksafeguard")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model dir names under outputs/remote_models (default: all with metrics)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default="outputs/remote_models")
    args = ap.parse_args()
    out = args.out or f"reports/{args.benchmark}/results/model_scores.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    tag = f"{args.benchmark}_evaluator_{args.evaluator}"
    rows, cat_rows = [], []
    names = args.models or sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                                  for p in glob.glob(f"{args.root}/*/{args.benchmark}/metrics_{tag}.json"))
    for name in names:
        d = f"{args.root}/{name}/{args.benchmark}"
        mp = f"{d}/metrics_{tag}.json"
        if not os.path.exists(mp):
            print(f"[skip] no metrics: {mp}")
            continue
        m = json.load(open(mp))
        k, x = m["k_idioms"], m["xstest"]
        rows.append({
            "model": name,
            "safe_compliance": m["safe_compliance_rate"],
            "safe_overrefusal": round(1 - m["safe_compliance_rate"], 4),
            "unsafe_refusal": m["unsafe_refusal_rate"],
            "behavior_acc": m["behavior_acc"],
            "refusal_f1": m["refusal_prf"]["f1"],
            "kidiom_comp": k["safe_compliance_rate"],
            "kidiom_refusal": k["unsafe_refusal_rate"],
            "harm_comp": x["safe_compliance_rate"],
            "harm_refusal": x["unsafe_refusal_rate"],
        })
        # per-category (mechanism) safe-compliance from results jsonl
        rp = f"{d}/results_{tag}.jsonl"
        if os.path.exists(rp):
            cnt = defaultdict(lambda: [0, 0]); trunc = 0; empty = 0; n = 0
            for line in open(rp, encoding="utf-8"):
                r = json.loads(line); n += 1
                if r.get("finish_reason") == "length": trunc += 1
                if not str(r.get("response", "")).strip(): empty += 1
                if r["label"] != "safe": continue
                lab = r["refusal_clf_label"]
                if lab not in ("compliance", "refusal", "deflection"): continue
                cnt[r["type"]][1] += 1
                if lab == "compliance": cnt[r["type"]][0] += 1
            cr = {"model": name, "n": n, "truncated": trunc, "empty_response": empty}
            for t, (c, tot) in sorted(cnt.items()):
                cr[t] = round(c / tot, 4) if tot else ""
            cat_rows.append(cr)

    rows.sort(key=lambda r: -r["safe_compliance"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} models)")
    if cat_rows:
        keys = ["model", "n", "truncated", "empty_response"] + sorted({k for r in cat_rows for k in r} - {"model", "n", "truncated", "empty_response"})
        base = os.path.basename(out).replace("model_scores", "model_category_scores")
        cat_out = os.path.join(os.path.dirname(out), base)
        with open(cat_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in sorted(cat_rows, key=lambda r: r["model"]): w.writerow({k: r.get(k, "") for k in keys})
        print(f"wrote {cat_out}")
    for r in rows:
        print(f"{r['model']:42s} comp={r['safe_compliance']:.3f} unsafe_ref={r['unsafe_refusal']:.3f} acc={r['behavior_acc']:.3f} f1={r['refusal_f1']:.3f} kid={r['kidiom_comp']:.2f}/{r['kidiom_refusal']:.2f} harm={r['harm_comp']:.2f}/{r['harm_refusal']:.2f}")


if __name__ == "__main__":
    main()
