# WildGuard-ko vs baseline evaluator — kanana-2-30b-a3b-instruct-2601

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.768 | 0.0533 | 0.8655 | 0.9231 | 0.8384 |
| k_idioms (ours) | 100 | 0.9 | 0.28 | 0.81 | 0.7627 | 0.8257 |
| xstest | 450 | 0.735 | 0.008 | 0.8778 | 0.9866 | 0.8424 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.06 | 0.05 |
| safe false-refusal ↓ | k_idioms (ours) | 0.22 | 0.2 |
| safe false-refusal ↓ | xstest | 0.028 | 0.02 |
| unsafe refusal / recall ↑ | overall | 0.78 | 0.7258 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.84 | 0.8 |
| unsafe refusal / recall ↑ | xstest | 0.765 | 0.7071 |
| refusal precision ↑ | overall | 0.9155 | 0.9231 |
| refusal precision ↑ | k_idioms (ours) | 0.7925 | 0.8 |
| refusal precision ↑ | xstest | 0.9563 | 0.9655 |
| refusal F1 ↑ | overall | 0.8423 | 0.8126 |
| refusal F1 ↑ | k_idioms (ours) | 0.8156 | 0.8 |
| refusal F1 ↑ | xstest | 0.85 | 0.8163 |
| behavior_acc ↑ (see note) | overall | 0.8673 | 0.8339 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.81 | 0.77 |
| behavior_acc ↑ (see note) | xstest | 0.88 | 0.8482 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0033 | 0.02 | 0 |
| k_idioms (ours) | 0.02 | 0.0 | 0 |
| xstest | 0.0 | 0.025 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **548**  |  agreement: **0.9599**  |  Cohen's κ: **0.914**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 192 | 19 |
| **WildGuard: non-refusal** | 3 | 334 |
