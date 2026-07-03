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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--skip_mm_profiling", action="store_true",
                        help="Skip vLLM's dummy-multimodal memory profiling pass. Needed for some "
                             "multimodal checkpoints (e.g. EXAONE-4.x) whose vision tower is incompatible "
                             "with the installed vLLM version and crashes during profiling. We only ever "
                             "send text, so this is a harmless no-op for text-only models.")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="xstest",
                        help="Benchmark name; used as a subdirectory so each model's outputs are "
                             "grouped per benchmark (outputs/remote_models/<model>/<benchmark>/). "
                             "Ignored when --save_dir is given explicitly.")

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
        args.save_dir = os.path.join("outputs/remote_models", model_name, args.benchmark)
    return args


# Models whose usable context we cap below their advertised window (e.g. for memory).
# Substring match against the model id.
_MODEL_MAX_LEN_CAPS = {
    "HyperCLOVAX": 4096,
}


def _config_max_pos(cfg) -> int:
    """Best-effort max context from a HF config (handles multimodal nested configs)."""
    for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
        if obj is None:
            continue
        v = getattr(obj, "max_position_embeddings", None)
        if isinstance(v, int) and v > 0:
            return v
    return 0


def derive_max_model_len(model_id: str, requested):
    """Clamp the requested max_model_len to what the model actually supports, so a
    single --max_model_len works across models with different context windows
    (vLLM errors if asked for more than the model's max_position_embeddings)."""
    if requested is None:
        return None
    caps = [requested]
    for key, cap in _MODEL_MAX_LEN_CAPS.items():
        if key.lower() in model_id.lower():
            caps.append(cap)
    try:
        from transformers import AutoConfig
        model_max = _config_max_pos(AutoConfig.from_pretrained(model_id, trust_remote_code=True))
        if model_max:
            caps.append(model_max)
    except Exception as e:
        print(f"[warn] could not read config for {model_id}: {e}; using requested max_model_len={requested}")
    eff = min(caps)
    if eff != requested:
        print(f"[info] max_model_len: requested {requested} -> {eff} (model limit) for {model_id}")
    return eff


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = build_engine(
        args.model, args.model_engine_backend,
        num_gpus=args.model_num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=derive_max_model_len(args.model, args.max_model_len),
        max_num_seqs=args.max_num_seqs,
        base_url=args.model_backend_base_url,
        skip_mm_profiling=args.skip_mm_profiling,
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

    inputs = [[instr] for instr in dataset[args.instruction_column]]
    responses = run_batch_generate(model, inputs, gen_params, args.batch_size, desc="Inference")

    results = [{**example, "response": resp} for example, resp in zip(dataset, responses)]
    output_dataset = Dataset.from_list(results)

    save_path = os.path.join(args.save_dir, "inference_outputs.jsonl")
    output_dataset.to_json(save_path, lines=True, force_ascii=False)
    print(f"Saved {len(results)} inference outputs → {save_path}")


if __name__ == "__main__":
    main()
