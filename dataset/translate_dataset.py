import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

FR_ROOT = Path(__file__).resolve().parents[1]
if str(FR_ROOT) not in sys.path:
    sys.path.insert(0, str(FR_ROOT))

from mllm import GenerationArgs, UniversalGenParams
from util import batched, build_engine, ensure_parent_dir
try:
    from generation_util import XSTEST_TRANSLATION_PROMPT, TRANSLATION_SYSTEM_PROMPT, TRANSLATION_FILTER_PROMPT
except ImportError:
    from dataset.generation_util import XSTEST_TRANSLATION_PROMPT, TRANSLATION_SYSTEM_PROMPT, TRANSLATION_FILTER_PROMPT


DEFAULT_SAVE_PATH = "/data/home/mk05/FR/dataset/xstest/xstest_ko.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--model_engine_backend", type=str, default="vllm")
    parser.add_argument("--model_backend_base_url", type=str, default=None)
    parser.add_argument("--model_num_gpus", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)

    parser.add_argument("--dataset_name", type=str, default="walledai/XSTest")
    parser.add_argument("--dataset_config", type=str, default=None,
                         help="Dataset config/subset name (e.g. 'or-bench-hard-1k' for bench-llm/or-bench). "
                              "Required for datasets that define multiple configs.")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--data_file", type=str, default=None,
                         help="Local file to translate instead of a Hub dataset (.csv / .json / .jsonl). "
                              "Takes precedence over --dataset_name when set.")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--filter_column", type=str, default=None,
                         help="If set, keep only rows whose value in this column is in --filter_values "
                              "(e.g. --filter_column Harmfulness --filter_values harmless for PHTest).")
    parser.add_argument("--filter_values", type=str, default="",
                         help="Comma-separated values to keep for --filter_column.")

    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_num_examples", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning", type=str, default="low")

    parser.add_argument("--translation_style", type=str, default="reasoning", choices=["reasoning", "plain"],
                        help="'reasoning' uses the think-then-[OUTPUT]: prompt as a single user message "
                             "(gpt-oss). 'plain' sends --system_prompt as a system message and the raw "
                             "English text as the user message, taking the output directly — the usage "
                             "expected by dedicated translators like nayohan/llama3-instrucTrans-enko-8b.")
    parser.add_argument("--system_prompt", type=str, default=TRANSLATION_SYSTEM_PROMPT,
                        help="System message used in --translation_style plain.")
    parser.add_argument("--translate_only", action="store_true",
                        help="Only translate and write every row to --save_path (no quality filtering, no "
                             "rejected split). Use this when filtering is done in a separate pass by a "
                             "different model (see filter_translations.py).")

    # ----- Translation quality filtering (mirrors the KS second-pass LLM check) -----
    parser.add_argument("--skip_llm_filter", action="store_true",
                         help="Skip the second-pass LLM check that flags unfaithful/broken translations. "
                              "Cheap heuristic checks (empty, no Hangul, refusal, etc.) always run.")
    parser.add_argument("--filter_batch_size", type=int, default=32,
                         help="Number of (source, translation) pairs judged per LLM filter call.")
    parser.add_argument("--filter_max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def _read_local_examples(path: str) -> List[Dict]:
    """Read a local .csv / .tsv / .json / .jsonl directly (no HF cache, which may be
    read-only). CSV is parsed with the csv module so quoted multi-line fields are
    handled correctly; cell values are whitespace-stripped."""
    ext = Path(path).suffix.lower()
    if ext in {".json", ".jsonl"}:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if ext == ".json" and text.startswith("["):
            return list(json.loads(text))
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    delimiter = "\t" if ext == ".tsv" else ","
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]


def load_examples(args) -> List[Dict]:
    if args.data_file:
        examples = _read_local_examples(args.data_file)
    elif args.dataset_config:
        examples = [dict(e) for e in load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)]
    else:
        examples = [dict(e) for e in load_dataset(args.dataset_name, split=args.dataset_split)]

    if examples and args.prompt_column not in examples[0]:
        raise ValueError(f"Column '{args.prompt_column}' not found. Available: {list(examples[0].keys())}")

    # Keep only rows whose --filter_column value is in --filter_values (e.g. only the
    # 'harmless' subset of PHTest's 'Harmfulness' column — pseudo-harmful = harmless).
    if args.filter_column:
        if examples and args.filter_column not in examples[0]:
            raise ValueError(f"Filter column '{args.filter_column}' not found. Available: {list(examples[0].keys())}")
        keep = {v.strip() for v in args.filter_values.split(",") if v.strip()}
        before = len(examples)
        examples = [e for e in examples if str(e.get(args.filter_column)).strip() in keep]
        print(f"[filter] {args.filter_column} in {sorted(keep)}: kept {len(examples)}/{before}")

    if args.max_num_examples is not None:
        examples = examples[:args.max_num_examples]

    return examples


_PUNCT = r"[.?!？。！]"


def _last_sentence_in_quotes(text: str) -> Optional[str]:
    """Last backtick- or double-quote-delimited sentence ending with punctuation."""
    for pattern in (rf"`([^`]{{5,}}{_PUNCT})`", rf'"([^"]{{5,}}{_PUNCT})"'):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def parse_translation(text: str) -> str:
    text = str(text or "").strip()
    # Last start-of-line [OUTPUT]: match — avoids matching format descriptions in thinking
    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate and not candidate.startswith("<") and len(candidate) > 3:
            return candidate
    # Fallback: last quoted sentence (backtick or double-quote) — handles truncation
    quoted = _last_sentence_in_quotes(text)
    if quoted:
        return quoted
    return text


def parse_plain_translation(text: str) -> str:
    """For dedicated translation models that emit the Korean translation directly,
    without a [OUTPUT]: tag. Strip whitespace and any wrapping quotes."""
    t = str(text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'`":
        t = t[1:-1].strip()
    return t


_HANGUL = re.compile(r"[가-힣]")
# Korean refusal/apology openers that mean the model refused instead of translating.
_REFUSAL_MARKERS = (
    "죄송", "도와드릴 수 없", "도와줄 수 없", "할 수 없습니다", "할 수 없어요",
    "as an ai", "i cannot", "i can't", "i'm sorry", "i am sorry", "sorry, but",
)


def heuristic_flags(source: str, translation: str) -> List[str]:
    """Cheap, model-free checks for obviously broken translations.
    Returns a list of flag strings; empty list means the translation looks fine."""
    flags: List[str] = []
    t = (translation or "").strip()
    s = (source or "").strip()

    if not t:
        return ["empty"]

    hangul = len(_HANGUL.findall(t))
    ascii_letters = sum(1 for c in t if c.isascii() and c.isalpha())
    if hangul == 0:
        flags.append("no_hangul")
    elif ascii_letters > hangul * 2:
        # Far more Latin letters than Hangul -> likely untranslated / code-switched.
        flags.append("mostly_english")

    if s and t == s:
        flags.append("identical_to_source")

    if s:
        ratio = len(t) / max(len(s), 1)
        if ratio < 0.2:
            flags.append("too_short")
        elif ratio > 5.0:
            flags.append("too_long")

    low = t.lower()
    if any(m.lower() in low for m in _REFUSAL_MARKERS):
        flags.append("looks_like_refusal")

    if "[OUTPUT]" in t or t.startswith("<"):
        flags.append("leftover_formatting")

    return flags


def build_filter_prompt(items: List[Dict]) -> str:
    """items: list of {"id": int, "source": str, "translation": str}."""
    block = "\n".join(
        f'{it["id"]}. [EN] {it["source"]}\n   [KO] {it["translation"]}' for it in items
    )
    return TRANSLATION_FILTER_PROMPT.format(items=block)


def parse_filter_result(raw_output: str) -> Dict[int, bool]:
    text = str(raw_output or "").strip()
    matches = re.findall(r"^\[OUTPUT\]:\s*(.+)", text, re.MULTILINE | re.DOTALL)
    candidate_text = matches[-1].strip() if matches else text

    json_match = re.search(r"\[.*\]", candidate_text, re.DOTALL)
    if not json_match:
        return {}
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {}

    verdicts: Dict[int, bool] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "id" in item and "ok" in item:
                try:
                    verdicts[int(item["id"])] = bool(item["ok"])
                except (TypeError, ValueError):
                    continue
    return verdicts


def llm_filter_translations(model, gen_params, items: List[Dict], batch_size: int) -> Dict[int, bool]:
    """Second-pass LLM fact-check: returns id -> ok. Conservative — any item the
    filter does not explicitly confirm as ok is treated as not-ok by the caller."""
    verdicts: Dict[int, bool] = {}
    if not items:
        return verdicts
    for batch in tqdm(list(batched(items, batch_size)), desc="Filtering"):
        prompt = build_filter_prompt(batch)
        gen_args = GenerationArgs(
            engine_input=[prompt],
            gen_params=gen_params,
            is_multi_turn_input=False,
            is_batch_input=True,
            add_generation_prompt=True,
        )
        outputs = model.generate(gen_args)
        raw = outputs[0].output_seqs[0] if outputs and outputs[0].output_seqs else ""
        verdicts.update(parse_filter_result(raw))
    return verdicts


def write_translations(examples: List[Dict], translations: List[str], save_path: str,
                       translation_column: str = "prompt_ko") -> None:
    """Write every translated row (no filtering) — used for --translate_only, when a
    separate pass filters the result with a different model."""
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        for example, translation in zip(examples, translations):
            record = dict(example)
            record[translation_column] = translation
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Done. Wrote {len(examples)} translations -> {save_path} (no filtering)")


def apply_translation_filter(model, examples: List[Dict], sources: List[str], translations: List[str],
                             save_path: str, *, skip_llm_filter: bool, filter_batch_size: int,
                             filter_max_new_tokens: int, seed: int = 0,
                             translation_column: str = "prompt_ko"):
    """Heuristic + (optional) LLM second-pass filter. Writes clean rows to save_path and
    flagged rows to <save_path>_rejected.jsonl; every record gets translation_ok and
    translation_flags. Shared by translate_dataset (single model) and filter_translations
    (separate filter model). Returns (kept, rejected)."""
    # Step 1: cheap heuristics on every row (empty, no Hangul, refusal, etc.).
    all_flags: List[List[str]] = [
        heuristic_flags(src, tr) for src, tr in zip(sources, translations)
    ]

    # Step 2: second-pass LLM check (KS-style) on rows that passed the heuristics,
    # to catch unfaithful-but-fluent translations the heuristics can't see.
    llm_ok: Dict[int, bool] = {}
    if not skip_llm_filter:
        to_check = [
            {"id": i, "source": sources[i], "translation": translations[i]}
            for i in range(len(examples))
            if not all_flags[i]
        ]
        filter_gen_params = UniversalGenParams(
            n=1,
            max_new_tokens=filter_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            reasoning="low",  # classification, not creative — keep it cheap/deterministic
        )
        llm_ok = llm_filter_translations(model, filter_gen_params, to_check, filter_batch_size)

    rejected_path = re.sub(r"(\.jsonl?|\.json)?$", "", save_path, count=1) + "_rejected.jsonl"
    ensure_parent_dir(rejected_path)

    kept = 0
    rejected = 0
    with open(save_path, "w", encoding="utf-8") as f_ok, \
         open(rejected_path, "w", encoding="utf-8") as f_bad:
        for i, (example, translation) in enumerate(zip(examples, translations)):
            flags = list(all_flags[i])
            if not skip_llm_filter and not flags:
                # Conservative: anything the filter didn't explicitly confirm is dropped.
                if llm_ok.get(i) is not True:
                    flags.append("llm_rejected")

            ok = len(flags) == 0
            record = dict(example)
            record[translation_column] = translation
            record["translation_ok"] = ok
            record["translation_flags"] = flags

            if ok:
                f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                f_bad.write(json.dumps(record, ensure_ascii=False) + "\n")
                rejected += 1

    print(f"Done. {len(examples)} rows -> {kept} kept ({save_path}), {rejected} rejected ({rejected_path}).")
    return kept, rejected


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    ensure_parent_dir(args.save_path)

    examples = load_examples(args)
    source_desc = args.data_file if args.data_file else f"{args.dataset_name} ({args.dataset_split} split)"
    print(f"Loaded {len(examples)} examples from {source_desc}")

    model = build_engine(
        args.model, args.model_engine_backend,
        num_gpus=args.model_num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        base_url=args.model_backend_base_url,
    )
    gen_params = UniversalGenParams(
        n=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        reasoning=args.reasoning,
    )

    sources = [str(ex.get(args.prompt_column) or "") for ex in examples]
    plain = args.translation_style == "plain"
    parse_fn = parse_plain_translation if plain else parse_translation
    if plain:
        # Dedicated translator: system message holds the instruction, user message is the raw text.
        engine_inputs = [[args.system_prompt, src] for src in sources]
    else:
        engine_inputs = [XSTEST_TRANSLATION_PROMPT.format(text=src) for src in sources]

    # mllm's batch preprocessor mishandles a length-1 batch for structured (system+user)
    # inputs, so never leave a trailing singleton batch.
    batches = list(batched(engine_inputs, args.batch_size))
    if plain and len(batches) >= 2 and len(batches[-1]) == 1:
        batches[-2] = batches[-2] + batches[-1]
        batches.pop()

    translations: List[str] = []
    for batch in tqdm(batches, desc="Translating"):
        gen_args = GenerationArgs(
            engine_input=batch,
            gen_params=gen_params,
            is_multi_turn_input=plain,
            is_batch_input=True,
            add_generation_prompt=True,
            use_system_prompt=plain,
        )
        outputs = model.generate(gen_args)
        for output in outputs:
            raw = output.output_seqs[0] if output.output_seqs else ""
            translations.append(parse_fn(raw))

    if args.translate_only:
        write_translations(examples, translations, args.save_path)
        return

    apply_translation_filter(
        model, examples, sources, translations, args.save_path,
        skip_llm_filter=args.skip_llm_filter,
        filter_batch_size=args.filter_batch_size,
        filter_max_new_tokens=args.filter_max_new_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
