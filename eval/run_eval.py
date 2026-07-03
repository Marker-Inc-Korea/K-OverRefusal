import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import UniversalGenParams
from util import build_engine, run_batch_generate

try:
    from eval.eval_util import (
        EVALUATORS, compute_metrics,
        compute_exaggerate_safety_metrics, compute_overrefusal_metrics, dump_metrics,
    )
except ImportError:
    from eval_util import (
        EVALUATORS, compute_metrics,
        compute_exaggerate_safety_metrics, compute_overrefusal_metrics, dump_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=str, default="openai/gpt-oss-safeguard-120b")
    parser.add_argument("--evaluator_type", type=str, default="auto",
                        choices=["auto", "llm_judge", "ksafeguard"],
                        help="Judge I/O type. 'llm_judge' = 3-way behavior-policy prompt "
                             "(gpt-oss/gpt-5.5/Llama-3.3...). 'ksafeguard' = WildGuard "
                             "moderation format, refusal/compliance. 'auto' picks ksafeguard "
                             "when the evaluator id contains 'ksafeguard'/'modguard'.")
    parser.add_argument("--evaluator_engine_backend", type=str, default="vllm")
    parser.add_argument("--evaluator_backend_base_url", type=str, default=None)
    parser.add_argument("--evaluator_num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to inference_outputs.jsonl produced by run_inference.py")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save results. Defaults to data_dir's parent directory.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Benchmark tag used in the output filenames "
                             "(results_<dataset>_evaluator_<evaluator>.jsonl). "
                             "Defaults to the save_dir basename.")
    parser.add_argument("--evaluator_name", type=str, default=None,
                        help="Evaluator tag used in the output filenames. Defaults to the "
                             "evaluator id leaf; set a clean name (e.g. 'ksafeguard') when the "
                             "evaluator is a long local path.")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--instruction_column", type=str, default="instruction")

    parser.add_argument("--metric", type=str, default="auto",
                        choices=["auto", "exaggerate", "overrefusal", "simple"],
                        help="Which metric report to compute. 'auto' picks 'exaggerate' when a 'label' "
                             "column exists (xstest), 'overrefusal' when a category column exists "
                             "(pseudo-harmful sets like OR-Bench), else 'simple'.")
    parser.add_argument("--category_column", type=str, default="category",
                        help="Column holding the prompt category, used for the per-category "
                             "over-refusal breakdown.")

    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(args.data_dir))
    if args.dataset_name is None:
        args.dataset_name = os.path.basename(os.path.normpath(args.save_dir))
    if args.evaluator_type == "auto":
        low = args.evaluator.lower()
        args.evaluator_type = "ksafeguard" if ("ksafeguard" in low or "modguard" in low) else "llm_judge"
    return args


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    evaluator = build_engine(
        args.evaluator, args.evaluator_engine_backend,
        num_gpus=args.evaluator_num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        base_url=args.evaluator_backend_base_url,
    )

    dataset = Dataset.from_json(args.data_dir)
    if args.max_num_examples is not None:
        dataset = dataset.select(range(min(args.max_num_examples, len(dataset))))
    print(f"Loaded {len(dataset)} examples from {args.data_dir}")

    format_input, parse_label = EVALUATORS[args.evaluator_type]
    print(f"Evaluator type: {args.evaluator_type}")
    eval_inputs = [
        format_input(ex[args.instruction_column], ex["response"])
        for ex in dataset
    ]

    gen_params = UniversalGenParams(n=1, max_new_tokens=args.max_new_tokens, temperature=0.0)
    clf_outputs = run_batch_generate(evaluator, eval_inputs, gen_params, args.batch_size,
                                     desc="Evaluating")
    labels = [parse_label(text) for text in clf_outputs]

    results = [
        {**example, "refusal_clf_output": clf_out, "refusal_clf_label": label}
        for example, clf_out, label in zip(dataset, clf_outputs, labels)
    ]
    output_dataset = Dataset.from_list(results)

    # Encode benchmark + evaluator in the filenames so multiple evaluators
    # (e.g. gpt-oss-120b, gpt-5.5, Llama-3.3-70B-Instruct) coexist without overwriting:
    #   results_<dataset>_evaluator_<evaluator>.jsonl / metrics_<dataset>_evaluator_<evaluator>.json
    evaluator_tag = args.evaluator_name or args.evaluator.split("/")[-1]
    tag = f"{args.dataset_name}_evaluator_{evaluator_tag}"
    results_save_path = os.path.join(args.save_dir, f"results_{tag}.jsonl")
    metrics_save_path = os.path.join(args.save_dir, f"metrics_{tag}.json")

    output_dataset.to_json(results_save_path, lines=True, force_ascii=False)

    cols = output_dataset.column_names
    if args.metric == "exaggerate" or (args.metric == "auto" and "label" in cols):
        metrics = compute_exaggerate_safety_metrics(output_dataset, example_type_key="label")
    elif args.metric == "overrefusal" or (args.metric == "auto" and args.category_column in cols):
        metrics = compute_overrefusal_metrics(output_dataset, category_key=args.category_column)
    else:
        metrics = compute_metrics(output_dataset)
    metrics_str = dump_metrics(metrics)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write(metrics_str + "\n")

    print(f"Saved evaluation results → {results_save_path}")
    print(f"Saved metrics → {metrics_save_path}")
    print(metrics_str)


if __name__ == "__main__":
    main()
