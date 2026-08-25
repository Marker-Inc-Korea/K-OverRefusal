"""Reasoning-block splitting shared by the vLLM engine (mllm.py) and the HF text runner.
Kept free of heavy imports so it can be used from a transformers-only environment."""
import re

_THINK_OPEN_RE = re.compile(r"<think>|<\|channel>thought\n?", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>|<channel\|>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"(?:<think>|<\|channel>thought\n?)(.*?)(?:</think>|<channel\|>)", re.DOTALL)


def split_reasoning(text: str, special_tokens=(), prompt_text: str = None):
    """Split a raw (special-token-preserving) completion into (final_answer, reasoning).
    Handles <think>...</think> (Qwen3.x / EXAONE-4.x / GLM-4.x / HyperCLOVAX-Think / kanana /
    Trillion / Motif) and gemma-4's <|channel>thought ... <channel|>. Thinking-on templates
    end the *prompt* with '<think>\n', so the completion may start inside the block and only
    the closer appears ('reasoning</think>answer'); pass prompt_text so a completion that ran
    out of tokens while still thinking is classified as reasoning, not answer. Remaining
    special tokens (eos etc.) are stripped from the answer."""
    if not text:
        return "", ""
    reasoning_parts = []
    out = text
    prompt_opened = bool(prompt_text) and prompt_text.rstrip().endswith("<think>")
    m = _THINK_CLOSE_RE.search(out)
    if m and not _THINK_OPEN_RE.search(out[:m.start()]):
        reasoning_parts.append(out[:m.start()])
        out = out[m.end():]
    elif prompt_opened and not m:
        reasoning_parts.append(out)
        out = ""

    def _grab(mm):
        reasoning_parts.append(mm.group(1))
        return ""
    out = _THINK_BLOCK_RE.sub(_grab, out)
    m = _THINK_OPEN_RE.search(out)
    if m:  # opened but never closed -> truncated inside the reasoning block
        reasoning_parts.append(out[m.end():])
        out = out[:m.start()]
    for t in sorted(set(special_tokens), key=len, reverse=True):
        if t and t in out:
            out = out.replace(t, "")
    reasoning = "\n".join(r.strip() for r in reasoning_parts if r.strip())
    return out.strip(), reasoning
