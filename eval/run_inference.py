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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning", type=str, default=None)

    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--instruction_column", type=str, default="instruction")

    args = parser.parse_args()
    if args.save_dir is None:
        model_name = args.model.rstrip("/").split("/")[-1]
        args.save_dir = os.path.join("outputs/remote_models", model_name)
    return args


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.model_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.max_model_len is not None:
            backend_kwargs["max_model_len"] = args.max_model_len
        if args.max_num_seqs is not None:
            backend_kwargs["max_num_seqs"] = args.max_num_seqs
    elif args.model_engine_backend in {"vllm-openai", "openrouter"}:
        backend_kwargs = {}
        if args.model_backend_base_url is not None:
            backend_kwargs["base_url"] = args.model_backend_base_url
    else:
        backend_kwargs = {}

    model = VLMInferenceEngine(
        args.model,
        backend=args.model_engine_backend,
        backend_kwargs=backend_kwargs,
    )

    dataset = Dataset.from_json(args.data_dir)
    if args.max_num_examples is not None:
        dataset = dataset.select(range(min(args.max_num_examples, len(dataset))))
    print(f"Loaded {len(dataset)} examples from {args.data_dir}")

    gen_params_kwargs = dict(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    if args.reasoning is not None:
        gen_params_kwargs["reasoning"] = args.reasoning
    gen_params = UniversalGenParams(**gen_params_kwargs)

    instructions = dataset[args.instruction_column]
    inputs = [[instr] for instr in instructions]

    responses = []
    for batch_inputs in tqdm(list(batched(inputs, args.batch_size)), desc="Inference"):
        gen_args = GenerationArgs(
            engine_input=batch_inputs,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        responses.extend(o.output_seqs[0] if o.output_seqs else "" for o in outputs)

    results = [{**example, "response": resp} for example, resp in zip(dataset, responses)]
    output_dataset = Dataset.from_list(results)

    save_path = os.path.join(args.save_dir, "inference_outputs.jsonl")
    output_dataset.to_json(save_path, lines=True, force_ascii=False)
    print(f"Saved {len(results)} inference outputs → {save_path}")


if __name__ == "__main__":
    main()
