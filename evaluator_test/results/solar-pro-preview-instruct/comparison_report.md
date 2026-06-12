# WildGuard-ko vs baseline evaluator — solar-pro-preview-instruct

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.764 | 0.0967 | 0.84 | 0.8682 | 0.8128 |
| k_idioms (ours) | 100 | 0.92 | 0.54 | 0.69 | 0.6301 | 0.7479 |
| xstest | 450 | 0.725 | 0.008 | 0.8733 | 0.9864 | 0.8357 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.2067 | 0.1467 |
| safe false-refusal ↓ | k_idioms (ours) | 0.42 | 0.42 |
| safe false-refusal ↓ | xstest | 0.164 | 0.092 |
| unsafe refusal / recall ↑ | overall | 0.652 | 0.576 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.64 | 0.66 |
| unsafe refusal / recall ↑ | xstest | 0.655 | 0.555 |
| refusal precision ↑ | overall | 0.7244 | 0.766 |
| refusal precision ↑ | k_idioms (ours) | 0.6038 | 0.6111 |
| refusal precision ↑ | xstest | 0.7616 | 0.8284 |
| refusal F1 ↑ | overall | 0.6863 | 0.6575 |
| refusal F1 ↑ | k_idioms (ours) | 0.6214 | 0.6346 |
| refusal F1 ↑ | xstest | 0.7043 | 0.6647 |
| behavior_acc ↑ (see note) | overall | 0.7291 | 0.6691 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.61 | 0.57 |
| behavior_acc ↑ (see note) | xstest | 0.7556 | 0.6911 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.0167 | 0.116 | 0 |
| k_idioms (ours) | 0.1 | 0.14 | 0 |
| xstest | 0.0 | 0.11 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9182**  |  Cohen's κ: **0.8264**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 184 | 41 |
| **WildGuard: non-refusal** | 4 | 321 |
