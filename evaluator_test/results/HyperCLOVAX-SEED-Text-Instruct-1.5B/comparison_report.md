# WildGuard-ko vs baseline evaluator — HyperCLOVAX-SEED-Text-Instruct-1.5B

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.78 | 0.0933 | 0.8491 | 0.8744 | 0.8245 |
| k_idioms (ours) | 100 | 0.94 | 0.52 | 0.71 | 0.6438 | 0.7642 |
| xstest | 450 | 0.74 | 0.008 | 0.88 | 0.9867 | 0.8457 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.15 | 0.12 |
| safe false-refusal ↓ | k_idioms (ours) | 0.48 | 0.4 |
| safe false-refusal ↓ | xstest | 0.084 | 0.064 |
| unsafe refusal / recall ↑ | overall | 0.672 | 0.648 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.78 | 0.76 |
| unsafe refusal / recall ↑ | xstest | 0.645 | 0.62 |
| refusal precision ↑ | overall | 0.7887 | 0.8182 |
| refusal precision ↑ | k_idioms (ours) | 0.619 | 0.6552 |
| refusal precision ↑ | xstest | 0.86 | 0.8857 |
| refusal F1 ↑ | overall | 0.7257 | 0.7232 |
| refusal F1 ↑ | k_idioms (ours) | 0.6902 | 0.7037 |
| refusal F1 ↑ | xstest | 0.7371 | 0.7294 |
| behavior_acc ↑ (see note) | overall | 0.7691 | 0.7564 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.65 | 0.61 |
| behavior_acc ↑ (see note) | xstest | 0.7956 | 0.7889 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.01 | 0.028 | 0 |
| k_idioms (ours) | 0.04 | 0.04 | 0 |
| xstest | 0.004 | 0.025 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9582**  |  Cohen's κ: **0.9107**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 194 | 19 |
| **WildGuard: non-refusal** | 4 | 333 |
