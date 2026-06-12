# WildGuard-ko vs baseline evaluator — llama-3-Korean-Bllossom-8B

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.704 | 0.0633 | 0.8309 | 0.9026 | 0.791 |
| k_idioms (ours) | 100 | 0.82 | 0.36 | 0.73 | 0.6949 | 0.7523 |
| xstest | 450 | 0.675 | 0.004 | 0.8533 | 0.9926 | 0.8036 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.0533 | 0.0468 |
| safe false-refusal ↓ | k_idioms (ours) | 0.22 | 0.1429 |
| safe false-refusal ↓ | xstest | 0.02 | 0.028 |
| unsafe refusal / recall ↑ | overall | 0.568 | 0.568 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.68 | 0.66 |
| unsafe refusal / recall ↑ | xstest | 0.54 | 0.545 |
| refusal precision ↑ | overall | 0.8987 | 0.9103 |
| refusal precision ↑ | k_idioms (ours) | 0.7556 | 0.825 |
| refusal precision ↑ | xstest | 0.9558 | 0.9397 |
| refusal F1 ↑ | overall | 0.6961 | 0.6995 |
| refusal F1 ↑ | k_idioms (ours) | 0.7158 | 0.7333 |
| refusal F1 ↑ | xstest | 0.6901 | 0.6899 |
| behavior_acc ↑ (see note) | overall | 0.7745 | 0.7395 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.73 | 0.6263 |
| behavior_acc ↑ (see note) | xstest | 0.7844 | 0.7644 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0133 | 0.024 | 0 |
| k_idioms (ours) | 0.06 | 0.0 | 0 |
| xstest | 0.004 | 0.03 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **549**  |  agreement: **0.949**  |  Cohen's κ: **0.8751**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 143 | 15 |
| **WildGuard: non-refusal** | 13 | 378 |
