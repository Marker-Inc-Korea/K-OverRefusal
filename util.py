"""Shared helpers for the K-OverRefusal pipeline (data generation + eval).

Kept intentionally small: jsonl IO, batching, and batched vLLM generation that
several stage scripts used to duplicate.
"""

import json
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def run_batch_generate(model, prompts: List[str], gen_params, batch_size: int) -> List[str]:
    """Batched greedy generation over a vLLM engine; returns one string per prompt."""
    from tqdm import tqdm
    from mllm import GenerationArgs
    results: List[str] = []
    for batch in tqdm(list(batched(prompts, batch_size)), desc="  LLM batch"):
        gen_args = GenerationArgs(
            engine_input=batch,
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        results.extend(o.output_seqs[0] if o.output_seqs else "" for o in outputs)
    return results
