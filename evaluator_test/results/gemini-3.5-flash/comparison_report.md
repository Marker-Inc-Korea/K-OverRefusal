# WildGuard-ko vs baseline evaluator — gemini-3.5-flash

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.72 | 0.0667 | 0.8364 | 0.9 | 0.8 |
| k_idioms (ours) | 100 | 0.8 | 0.38 | 0.71 | 0.678 | 0.734 |
| xstest | 450 | 0.7 | 0.004 | 0.8644 | 0.9929 | 0.8211 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.01 | 0.0033 |
| safe false-refusal ↓ | k_idioms (ours) | 0.0 | 0.0 |
| safe false-refusal ↓ | xstest | 0.012 | 0.004 |
| unsafe refusal / recall ↑ | overall | 0.62 | 0.624 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.56 | 0.56 |
| unsafe refusal / recall ↑ | xstest | 0.635 | 0.64 |
| refusal precision ↑ | overall | 0.981 | 0.9936 |
| refusal precision ↑ | k_idioms (ours) | 1.0 | 1.0 |
| refusal precision ↑ | xstest | 0.9769 | 0.9922 |
| refusal F1 ↑ | overall | 0.7598 | 0.7666 |
| refusal F1 ↑ | k_idioms (ours) | 0.7179 | 0.7179 |
| refusal F1 ↑ | xstest | 0.7697 | 0.7781 |
| behavior_acc ↑ (see note) | overall | 0.8218 | 0.8218 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.78 | 0.78 |
| behavior_acc ↑ (see note) | xstest | 0.8311 | 0.8311 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.04 | 0.056 | 0 |
| k_idioms (ours) | 0.24 | 0.1 | 0 |
| xstest | 0.0 | 0.045 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9727**  |  Cohen's κ: **0.9333**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 150 | 8 |
| **WildGuard: non-refusal** | 7 | 385 |
