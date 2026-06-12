# WildGuard-ko vs baseline evaluator — A.X-4.0-Light

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.748 | 0.0967 | 0.8327 | 0.8657 | 0.8026 |
| k_idioms (ours) | 100 | 0.92 | 0.52 | 0.7 | 0.6389 | 0.7541 |
| xstest | 450 | 0.705 | 0.012 | 0.8622 | 0.9792 | 0.8198 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.1133 | 0.11 |
| safe false-refusal ↓ | k_idioms (ours) | 0.44 | 0.44 |
| safe false-refusal ↓ | xstest | 0.048 | 0.044 |
| unsafe refusal / recall ↑ | overall | 0.72 | 0.712 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.76 | 0.74 |
| unsafe refusal / recall ↑ | xstest | 0.71 | 0.705 |
| refusal precision ↑ | overall | 0.8411 | 0.8436 |
| refusal precision ↑ | k_idioms (ours) | 0.6333 | 0.6271 |
| refusal precision ↑ | xstest | 0.9221 | 0.9276 |
| refusal F1 ↑ | overall | 0.7759 | 0.7722 |
| refusal F1 ↑ | k_idioms (ours) | 0.6909 | 0.6789 |
| refusal F1 ↑ | xstest | 0.8023 | 0.8011 |
| behavior_acc ↑ (see note) | overall | 0.8109 | 0.8 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.66 | 0.62 |
| behavior_acc ↑ (see note) | xstest | 0.8444 | 0.84 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0167 | 0.02 | 0 |
| k_idioms (ours) | 0.04 | 0.08 | 0 |
| xstest | 0.012 | 0.005 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9836**  |  Cohen's κ: **0.9655**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 208 | 6 |
| **WildGuard: non-refusal** | 3 | 333 |
