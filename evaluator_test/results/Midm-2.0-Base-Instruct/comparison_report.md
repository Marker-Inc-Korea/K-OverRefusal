# WildGuard-ko vs baseline evaluator — Midm-2.0-Base-Instruct

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.8 | 0.0867 | 0.8618 | 0.885 | 0.8404 |
| k_idioms (ours) | 100 | 0.94 | 0.5 | 0.72 | 0.6528 | 0.7705 |
| xstest | 450 | 0.765 | 0.004 | 0.8933 | 0.9935 | 0.8644 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.1367 | 0.1171 |
| safe false-refusal ↓ | k_idioms (ours) | 0.68 | 0.62 |
| safe false-refusal ↓ | xstest | 0.028 | 0.0161 |
| unsafe refusal / recall ↑ | overall | 0.88 | 0.844 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.94 | 0.92 |
| unsafe refusal / recall ↑ | xstest | 0.865 | 0.825 |
| refusal precision ↑ | overall | 0.8429 | 0.8577 |
| refusal precision ↑ | k_idioms (ours) | 0.5802 | 0.5974 |
| refusal precision ↑ | xstest | 0.9611 | 0.9763 |
| refusal F1 ↑ | overall | 0.8611 | 0.8508 |
| refusal F1 ↑ | k_idioms (ours) | 0.7175 | 0.7244 |
| refusal F1 ↑ | xstest | 0.9105 | 0.8943 |
| behavior_acc ↑ (see note) | overall | 0.8709 | 0.8579 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.63 | 0.64 |
| behavior_acc ↑ (see note) | xstest | 0.9244 | 0.9065 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0033 | 0.008 | 0 |
| k_idioms (ours) | 0.0 | 0.0 | 0 |
| xstest | 0.004 | 0.01 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **549**  |  agreement: **0.9636**  |  Cohen's κ: **0.9267**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 243 | 17 |
| **WildGuard: non-refusal** | 3 | 286 |
