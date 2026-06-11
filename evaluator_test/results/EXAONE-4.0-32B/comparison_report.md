# WildGuard-ko vs baseline evaluator — EXAONE-4.0-32B

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.716 | 0.0667 | 0.8345 | 0.8995 | 0.7973 |
| k_idioms (ours) | 100 | 0.84 | 0.34 | 0.75 | 0.7119 | 0.7707 |
| xstest | 450 | 0.685 | 0.012 | 0.8533 | 0.9786 | 0.8059 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.0467 | 0.0567 |
| safe false-refusal ↓ | k_idioms (ours) | 0.2 | 0.18 |
| safe false-refusal ↓ | xstest | 0.016 | 0.032 |
| unsafe refusal / recall ↑ | overall | 0.656 | 0.5847 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.7 | 0.6 |
| unsafe refusal / recall ↑ | xstest | 0.645 | 0.5808 |
| refusal precision ↑ | overall | 0.9213 | 0.8951 |
| refusal precision ↑ | k_idioms (ours) | 0.7778 | 0.7692 |
| refusal precision ↑ | xstest | 0.9699 | 0.935 |
| refusal F1 ↑ | overall | 0.7663 | 0.7073 |
| refusal F1 ↑ | k_idioms (ours) | 0.7369 | 0.6742 |
| refusal F1 ↑ | xstest | 0.7748 | 0.7165 |
| behavior_acc ↑ (see note) | overall | 0.8182 | 0.7737 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.75 | 0.68 |
| behavior_acc ↑ (see note) | xstest | 0.8333 | 0.7946 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0167 | 0.032 | 0 |
| k_idioms (ours) | 0.06 | 0.1 | 0 |
| xstest | 0.008 | 0.015 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **548**  |  agreement: **0.9215**  |  Cohen's κ: **0.8165**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 148 | 29 |
| **WildGuard: non-refusal** | 14 | 357 |
