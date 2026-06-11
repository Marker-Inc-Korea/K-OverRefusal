# WildGuard-ko vs baseline evaluator — gpt-5.5

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.736 | 0.0833 | 0.8345 | 0.8804 | 0.8018 |
| k_idioms (ours) | 100 | 0.88 | 0.44 | 0.72 | 0.6667 | 0.7586 |
| xstest | 450 | 0.7 | 0.012 | 0.86 | 0.979 | 0.8163 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.05 | 0.0567 |
| safe false-refusal ↓ | k_idioms (ours) | 0.08 | 0.16 |
| safe false-refusal ↓ | xstest | 0.044 | 0.036 |
| unsafe refusal / recall ↑ | overall | 0.788 | 0.772 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.84 | 0.88 |
| unsafe refusal / recall ↑ | xstest | 0.775 | 0.745 |
| refusal precision ↑ | overall | 0.9292 | 0.919 |
| refusal precision ↑ | k_idioms (ours) | 0.913 | 0.8462 |
| refusal precision ↑ | xstest | 0.9337 | 0.943 |
| refusal F1 ↑ | overall | 0.8528 | 0.8391 |
| refusal F1 ↑ | k_idioms (ours) | 0.875 | 0.8627 |
| refusal F1 ↑ | xstest | 0.847 | 0.8324 |
| behavior_acc ↑ (see note) | overall | 0.8764 | 0.8545 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.88 | 0.84 |
| behavior_acc ↑ (see note) | xstest | 0.8756 | 0.8578 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0167 | 0.028 | 0 |
| k_idioms (ours) | 0.06 | 0.02 | 0 |
| xstest | 0.008 | 0.03 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9455**  |  Cohen's κ: **0.8847**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 196 | 16 |
| **WildGuard: non-refusal** | 14 | 324 |
