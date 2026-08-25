"""HF-transformers text runner for checkpoints whose architecture vLLM does not know
(custom remote code: NC AI VAETKI, Motif, Trillion Tri-*, ...). Produces the same
inference_outputs.jsonl schema as run_inference.py (response / reasoning / finish_reason)
so run_eval.py and the report scripts need no changes."""
import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from eval.reasoning_utils import split_reasoning  # noqa: E402 (no vLLM dependency)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--instruction_column", default="prompt_ko")
    p.add_argument("--benchmark", default="merged_v2")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--thinking", default="off", choices=["off", "on", "default"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa | eager | flash_attention_2)")
    p.add_argument("--max_num_examples", type=int, default=None)
    a = p.parse_args()
    if a.save_dir is None:
        a.save_dir = os.path.join("outputs/remote_models", a.model.rstrip("/").split("/")[-1], a.benchmark)
    return a


def _compat_config(model_id):
    """Remote-code checkpoints written for transformers 4.5x (e.g. NC AI VAETKI) break on
    5.x in two places: ROPE_INIT_FUNCTIONS no longer has a 'default' entry, and a null
    `rope_scaling` is normalised to {'rope_type': 'default', ...} which their code then
    indexes for 'original_max_position_embeddings'. Restore both behaviours."""
    import torch
    from transformers import AutoConfig
    from transformers import modeling_rope_utils as mru

    def _default_rope(config, device=None, seq_len=None, layer_type=None):
        base = getattr(config, "rope_theta", None) or (getattr(config, "rope_parameters", {}) or {}).get("rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0
    mru.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)

    # transformers 5.x renamed create_causal_mask(input_embeds=...) -> inputs_embeds and dropped
    # cache_position; 4.5x-era remote code still passes the old kwargs. Wrap the functions
    # BEFORE the remote modeling module is imported (it binds the names at import time).
    import inspect
    from transformers import masking_utils as mu
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(mu, name, None)
        if fn is None or getattr(fn, "_fr_compat", False):
            continue
        accepted = set(inspect.signature(fn).parameters)
        if "input_embeds" in accepted:
            continue  # old API, nothing to do

        def _make(fn, accepted):
            def wrapper(*args, **kwargs):
                if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
                return fn(*args, **kwargs)
            wrapper._fr_compat = True
            return wrapper
        setattr(mu, name, _make(fn, accepted))
    print("[compat] masking_utils kwargs shim installed")

    # transformers 5.x dropped "eager" from ALL_ATTENTION_FUNCTIONS (models now carry their
    # own eager fn); 4.5x-era remote code indexes the registry directly. Register a plain
    # softmax attention under "eager" (GQA-aware, bool or additive masks, v_dim != qk_dim ok).
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "eager" not in ALL_ATTENTION_FUNCTIONS:
        import torch.nn.functional as F

        def _eager_attention_forward(module, query, key, value, attention_mask, scaling=None,
                                     dropout=0.0, **kwargs):
            n_rep = query.shape[1] // key.shape[1]
            if n_rep > 1:
                key = key.repeat_interleave(n_rep, dim=1)
                value = value.repeat_interleave(n_rep, dim=1)
            if scaling is None:
                scaling = query.shape[-1] ** -0.5
            attn = torch.matmul(query, key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                m = attention_mask[:, :, :, : key.shape[-2]]
                attn = attn.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else attn + m
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
            attn = F.dropout(attn, p=dropout, training=module.training)
            out = torch.matmul(attn, value).transpose(1, 2).contiguous()
            return out, attn
        ALL_ATTENTION_FUNCTIONS["eager"] = _eager_attention_forward
        print("[compat] registered generic eager attention in ALL_ATTENTION_FUNCTIONS")

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and rs.get("rope_type", "default") == "default" \
            and "original_max_position_embeddings" not in rs:
        try:
            cfg.rope_scaling = None
            print("[compat] rope_scaling normalised dict -> None (transformers 5.x shim)")
        except Exception as e:
            print(f"[compat] could not reset rope_scaling: {e}")
    return cfg


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = _compat_config(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=args.attn)
    except Exception as e:
        print(f"[warn] load with attn={args.attn} failed ({type(e).__name__}: {str(e)[:200]}); retrying eager")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager")
    model.eval()
    print(f"[info] loaded {args.model}: {type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params, "
          f"gpu mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    ds = Dataset.from_json(args.data_dir)
    if args.max_num_examples is not None:
        ds = ds.select(range(min(args.max_num_examples, len(ds))))
    print(f"Loaded {len(ds)} examples from {args.data_dir}")

    ct_kwargs = {}
    if args.thinking == "off":
        ct_kwargs["enable_thinking"] = False
    elif args.thinking == "on":
        ct_kwargs["enable_thinking"] = True

    def render(instr):
        msgs = [{"role": "user", "content": instr}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **ct_kwargs)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [render(x) for x in ds[args.instruction_column]]
    print(f"[info] thinking={args.thinking} chat_template_kwargs={ct_kwargs}; example prompt tail: "
          f"{prompts[0][-120:]!r}")

    # generate in length-sorted batches (less padding), then restore the original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    special = list(getattr(tok, "all_special_tokens", []) or [])
    eos_ids = set()
    for e in (getattr(model.generation_config, "eos_token_id", None), tok.eos_token_id):
        if isinstance(e, int):
            eos_ids.add(e)
        elif isinstance(e, (list, tuple)):
            eos_ids.update(int(x) for x in e)
    for start in tqdm(range(0, len(order), args.batch_size), desc="HF inference"):
        idxs = order[start:start + args.batch_size]
        batch = [prompts[i] for i in idxs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(model.device)
        enc.pop("token_type_ids", None)  # some tokenizers still emit it; generate() rejects unused kwargs
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        for row, i in enumerate(idxs):
            ids = new_tokens[row].tolist()
            # strip right padding
            while ids and ids[-1] == tok.pad_token_id and tok.pad_token_id not in eos_ids:
                ids.pop()
            hit_eos = any(t in eos_ids for t in ids)
            finish = "stop" if hit_eos else ("length" if len(ids) >= args.max_new_tokens else "stop")
            raw = tok.decode(ids, skip_special_tokens=False)
            resp, reasoning = split_reasoning(raw, special, prompt_text=prompts[i])
            results[i] = {**ds[i], "response": resp, "reasoning": reasoning, "finish_reason": finish}

    out = Dataset.from_list(results)
    n_reason = sum(1 for r in results if r["reasoning"])
    n_trunc = sum(1 for r in results if r["finish_reason"] == "length")
    n_empty = sum(1 for r in results if not r["response"].strip())
    print(f"[info] outputs: {len(results)} | with reasoning block: {n_reason} | truncated: {n_trunc} | empty: {n_empty}")
    save_path = os.path.join(args.save_dir, "inference_outputs.jsonl")
    out.to_json(save_path, lines=True, force_ascii=False)
    print(f"Saved {len(results)} inference outputs → {save_path}")


if __name__ == "__main__":
    main()
