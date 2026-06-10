import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import VLMInferenceEngine, UniversalGenParams, GenerationArgs

try:
    from eval.eval_util import format_eval_input, get_label_str, compute_metrics, compute_exaggerate_safety_metrics, dump_metrics
except ImportError:
    from eval_util import format_eval_input, get_label_str, compute_metrics, compute_exaggerate_safety_metrics, dump_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=str, default="openai/gpt-oss-safeguard-120b")
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

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--instruction_column", type=str, default="instruction")

    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(args.data_dir))
    return args


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.evaluator_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.evaluator_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.max_model_len is not None:
            backend_kwargs["max_model_len"] = args.max_model_len
        if args.max_num_seqs is not None:
            backend_kwargs["max_num_seqs"] = args.max_num_seqs
    elif args.evaluator_engine_backend in {"vllm-openai", "openrouter"}:
        backend_kwargs = {}
        if args.evaluator_backend_base_url is not None:
            backend_kwargs["base_url"] = args.evaluator_backend_base_url
    else:
        backend_kwargs = {}

    evaluator = VLMInferenceEngine(
        args.evaluator,
        backend=args.evaluator_engine_backend,
        backend_kwargs=backend_kwargs,
    )

    dataset = Dataset.from_json(args.data_dir)
    if args.max_num_examples is not None:
        dataset = dataset.select(range(min(args.max_num_examples, len(dataset))))
    print(f"Loaded {len(dataset)} examples from {args.data_dir}")

    eval_inputs = [
        format_eval_input(ex[args.instruction_column], ex["response"])
        for ex in dataset
    ]

    gen_params = UniversalGenParams(n=1, max_new_tokens=args.max_new_tokens, temperature=0.0)

    clf_outputs = []
    for batch_inputs in tqdm(list(batched(eval_inputs, args.batch_size)), desc="Evaluating"):
        gen_args = GenerationArgs(
            engine_input=batch_inputs,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = evaluator.generate(gen_args)
        clf_outputs.extend(o.output_seqs[0] if o.output_seqs else "" for o in outputs)

    labels = [get_label_str(text) for text in clf_outputs]

    results = [
        {**example, "refusal_clf_output": clf_out, "refusal_clf_label": label}
        for example, clf_out, label in zip(dataset, clf_outputs, labels)
    ]
    output_dataset = Dataset.from_list(results)

    results_save_path = os.path.join(args.save_dir, "evaluation_results.jsonl")
    metrics_save_path = os.path.join(args.save_dir, "metrics.json")

    output_dataset.to_json(results_save_path, lines=True, force_ascii=False)

    label_key = "label"
    if label_key in output_dataset.column_names:
        metrics = compute_exaggerate_safety_metrics(output_dataset, example_type_key=label_key)
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
