"""Compute inter-judge agreement between two evaluators that both labelled the
same model responses with the 3-class scheme {refusal, deflection, compliance}.

Default comparison:
    Judge A = gpt-oss-safeguard-120b   -> outputs/remote_models/<model>/evaluation_results.jsonl
    Judge B = gpt-5.5 (openrouter)     -> outputs/cross_judge/<model>/evaluation_results.jsonl

Both files come from the SAME inference_outputs.jsonl (same order), and each row
carries `refusal_clf_label`. We align row-by-row and report raw agreement,
Cohen's kappa (3-class), a 3x3 confusion matrix, and a collapsed binary-refusal
agreement -- overall and per group (k_idioms = our hand-made split, reported
independently from xstest).

The metric JSON (and a small markdown summary) is written under this directory
(evaluator_test/judge_agreement/).
"""

import argparse
import json
import os
import sys
from pathlib import Path

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

EVALUATOR_TEST_DIR = Path(__file__).resolve().parent

try:
    from eval.eval_util import K_IDIOMS_TYPES
except ImportError:
    from eval_util import K_IDIOMS_TYPES

LABELS = ["refusal", "deflection", "compliance"]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _r(x):
    return round(x, 4)


def _kappa(pairs, classes):
    """Cohen's kappa over a list of (a, b) label tuples for the given class set."""
    n = len(pairs)
    if n == 0:
        return 0.0
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    conf = [[0] * k for _ in range(k)]
    for a, b in pairs:
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) / n for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0


def _to_binary(label):
    return "refusal" if label == "refusal" else "non_refusal"


def compute_pair_metrics(pairs):
    """pairs: list of (judge_a_label, judge_b_label), both in LABELS."""
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    # 3x3 confusion: rows = judge A, cols = judge B
    conf = {ra: {cb: 0 for cb in LABELS} for ra in LABELS}
    for a, b in pairs:
        conf[a][b] += 1

    bin_pairs = [(_to_binary(a), _to_binary(b)) for a, b in pairs]
    bin_agree = sum(1 for a, b in bin_pairs if a == b)

    return {
        "n": n,
        "raw_agreement": _r(agree / n) if n else 0.0,
        "cohen_kappa": _r(_kappa(pairs, LABELS)) if n else 0.0,
        "binary_refusal_agreement": _r(bin_agree / n) if n else 0.0,
        "binary_refusal_kappa": _r(_kappa(bin_pairs, ["refusal", "non_refusal"])) if n else 0.0,
        "confusion": conf,
    }


def collect_pairs(rows_a, rows_b, label_column):
    """Align two result files row-by-row; return (pairs, by_group, misaligned)."""
    n = min(len(rows_a), len(rows_b))
    pairs_all, pairs_k, pairs_x = [], [], []
    misaligned = 0
    skipped = 0
    for i in range(n):
        ra, rb = rows_a[i], rows_b[i]
        if str(ra.get("response", ""))[:60] != str(rb.get("response", ""))[:60]:
            misaligned += 1
        la, lb = ra.get(label_column), rb.get(label_column)
        if la not in LABELS or lb not in LABELS:
            skipped += 1
            continue
        pairs_all.append((la, lb))
        if ra.get("type") in K_IDIOMS_TYPES:
            pairs_k.append((la, lb))
        else:
            pairs_x.append((la, lb))
    return pairs_all, pairs_k, pairs_x, misaligned, skipped


def model_metrics(pairs_all, pairs_k, pairs_x, misaligned, skipped):
    m = compute_pair_metrics(pairs_all)
    m["k_idioms"] = compute_pair_metrics(pairs_k)
    m["xstest"] = compute_pair_metrics(pairs_x)
    m["misaligned_rows"] = misaligned
    m["skipped_unlabeled"] = skipped
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--safeguard_dir", type=str,
                   default=str(FR_ROOT / "outputs" / "remote_models"),
                   help="Dir with <model>/evaluation_results.jsonl from judge A (gpt-oss-safeguard-120b).")
    p.add_argument("--gpt5_dir", type=str,
                   default=str(FR_ROOT / "outputs" / "cross_judge"),
                   help="Dir with <model>/evaluation_results.jsonl from judge B (gpt-5.5).")
    p.add_argument("--save_dir", type=str,
                   default=str(EVALUATOR_TEST_DIR / "judge_agreement"))
    p.add_argument("--label_column", type=str, default="refusal_clf_label")
    p.add_argument("--judge_a_name", type=str, default="gpt-oss-safeguard-120b")
    p.add_argument("--judge_b_name", type=str, default="gpt-5.5")
    p.add_argument("--results_filename", type=str, default="evaluation_results.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    safeguard_dir = Path(args.safeguard_dir)
    gpt5_dir = Path(args.gpt5_dir)

    # models present in BOTH judge dirs
    a_models = {d.name for d in safeguard_dir.iterdir()
                if (d / args.results_filename).is_file()} if safeguard_dir.is_dir() else set()
    b_models = {d.name for d in gpt5_dir.iterdir()
                if (d / args.results_filename).is_file()} if gpt5_dir.is_dir() else set()
    common = sorted(a_models & b_models)
    missing = sorted((a_models | b_models) - set(common))

    if not common:
        print(f"[WARN] No models found in BOTH:\n  A: {safeguard_dir}\n  B: {gpt5_dir}")
        print(f"  A has: {sorted(a_models)}\n  B has: {sorted(b_models)}")
        print("Run the gpt-5.5 eval with --save_dir pointing to the gpt5_dir first.")

    per_model = {}
    overall_all, overall_k, overall_x = [], [], []
    total_misaligned = total_skipped = 0

    for name in common:
        rows_a = _read_jsonl(safeguard_dir / name / args.results_filename)
        rows_b = _read_jsonl(gpt5_dir / name / args.results_filename)
        pa, pk, px, mis, skp = collect_pairs(rows_a, rows_b, args.label_column)
        per_model[name] = model_metrics(pa, pk, px, mis, skp)
        overall_all += pa
        overall_k += pk
        overall_x += px
        total_misaligned += mis
        total_skipped += skp
        print(f"[{name}] n={len(pa)}  raw={per_model[name]['raw_agreement']}  "
              f"kappa={per_model[name]['cohen_kappa']}  "
              f"binary={per_model[name]['binary_refusal_agreement']}")

    metrics = {
        "judge_a": args.judge_a_name,
        "judge_b": args.judge_b_name,
        "label_column": args.label_column,
        "models_compared": common,
        "models_missing_one_side": missing,
        "overall": model_metrics(overall_all, overall_k, overall_x, total_misaligned, total_skipped),
        "per_model": per_model,
    }

    metrics_path = os.path.join(args.save_dir, "agreement_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics → {metrics_path}")

    report_path = os.path.join(args.save_dir, "agreement_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(metrics))
    print(f"Saved report  → {report_path}")


def build_report(metrics) -> str:
    a, b = metrics["judge_a"], metrics["judge_b"]
    L = [f"# Inter-judge agreement — {a} vs {b}\n",
         f"- Label scheme: 3-class {{refusal, deflection, compliance}} on `{metrics['label_column']}`",
         f"- Models compared: {', '.join(metrics['models_compared']) or '(none)'}"]
    if metrics["models_missing_one_side"]:
        L.append(f"- ⚠️ Present on only one side (skipped): {', '.join(metrics['models_missing_one_side'])}")
    L.append("")

    def row(name, m):
        return (f"| {name} | {m['n']} | {m['raw_agreement']} | {m['cohen_kappa']} | "
                f"{m['binary_refusal_agreement']} | {m['binary_refusal_kappa']} |")

    for grp_label, grp_key in [("Overall (all groups)", None),
                               ("k_idioms (ours)", "k_idioms"),
                               ("xstest", "xstest")]:
        L.append(f"## {grp_label}\n")
        L.append("| Model | n | 3-class agree | 3-class κ | binary-refusal agree | binary-refusal κ |")
        L.append("|---|---|---|---|---|---|")
        ov = metrics["overall"] if grp_key is None else metrics["overall"].get(grp_key, {})
        L.append(row("**ALL MODELS**", ov))
        for name, m in metrics["per_model"].items():
            mm = m if grp_key is None else m.get(grp_key, {})
            L.append(row(name, mm))
        L.append("")

    # confusion matrix (overall, all groups)
    conf = metrics["overall"]["confusion"]
    L.append("## Overall 3x3 confusion (rows = " + a + ", cols = " + b + ")\n")
    L.append("| A \\ B | refusal | deflection | compliance |")
    L.append("|---|---|---|---|")
    for ra in LABELS:
        L.append(f"| **{ra}** | {conf[ra]['refusal']} | {conf[ra]['deflection']} | {conf[ra]['compliance']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
