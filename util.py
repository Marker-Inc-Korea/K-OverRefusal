"""Shared helpers for the K-OverRefusal pipeline (data generation + eval):
jsonl IO, batching, engine construction, and batched generation."""

import json
import os
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def build_backend_kwargs(backend: str, num_gpus: int = 1, gpu_memory_utilization: float = 0.85,
                         max_model_len=None, max_num_seqs=None, base_url=None,
                         skip_mm_profiling: bool = False, quantization=None,
                         moe_backend=None) -> Dict:
    if backend == "vllm":
        kw = {"tensor_parallel_size": num_gpus,
              "gpu_memory_utilization": gpu_memory_utilization}
        if max_model_len is not None:
            kw["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            kw["max_num_seqs"] = max_num_seqs
        if skip_mm_profiling:
            kw["skip_mm_profiling"] = True
        if quantization:  # e.g. "fp8" = on-the-fly weight quantization (70B-class on one 96GB GPU)
            kw["quantization"] = quantization
        if moe_backend:   # e.g. "triton" — the portable MoE kernel on SM 12.0 (Blackwell workstation)
            kw["moe_backend"] = moe_backend
        return kw
    if backend in {"vllm-openai", "openrouter"}:
        return {"base_url": base_url} if base_url is not None else {}
    return {}


def build_engine(model_id: str, backend: str = "vllm", chat_template_kwargs=None,
                 strip_reasoning: bool = True, **backend_options):
    """Construct a VLMInferenceEngine; backend_options as in build_backend_kwargs.
    chat_template_kwargs (e.g. {"enable_thinking": False}) are forwarded to the chat
    template; strip_reasoning splits <think>-style blocks off vLLM completions."""
    from mllm import VLMInferenceEngine
    return VLMInferenceEngine(model_id, backend=backend,
                              backend_kwargs=build_backend_kwargs(backend, **backend_options),
                              chat_template_kwargs=chat_template_kwargs,
                              strip_reasoning=strip_reasoning)


def run_batch_generate(model, prompts: List, gen_params, batch_size: int,
                       desc: str = "  LLM batch", return_meta: bool = False):
    """Batched generation over a vLLM engine; returns one string per prompt.
    With return_meta=True returns one dict per prompt instead:
    {"response", "reasoning" (split-off chain-of-thought, "" if none), "finish_reason"}."""
    from tqdm import tqdm
    from mllm import GenerationArgs
    results = []
    for batch in tqdm(list(batched(prompts, batch_size)), desc=desc):
        gen_args = GenerationArgs(
            engine_input=batch,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        for o in outputs:
            text = o.output_seqs[0] if o.output_seqs else ""
            if not return_meta:
                results.append(text)
                continue
            reasoning = getattr(o, "reasoning_seqs", None) or [""]
            finish = getattr(o, "finish_reasons", None) or [None]
            results.append({"response": text, "reasoning": reasoning[0] or "",
                            "finish_reason": finish[0]})
    return results
