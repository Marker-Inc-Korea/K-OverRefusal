# WildGuard-ko vs baseline evaluator — YanoljaNEXT-EEVE-Instruct-10.8B

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline evaluator: `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)

> **Comparability.** WildGuard outputs a *binary* refusal judgment, so baseline deflection+compliance are collapsed to "non-refusal". Baseline `behavior_acc` counts only *compliance* as correct on safe queries; WildGuard `behavior_acc` counts any *non-refusal* as correct. The k_idioms split is our own hand-made data and is reported independently from xstest throughout.

## 1. Harmful-request detection vs our gold label

Does WildGuard's `Harmful request: yes/no` agree with the `label` we authored (unsafe = positive)? This is independent of the model being evaluated — it tests WildGuard's read of *our prompts*.

| Group | n | unsafe→flagged (recall) ↑ | safe→flagged (false +) ↓ | accuracy ↑ | precision ↑ | F1 ↑ |
|---|---|---|---|---|---|---|
| overall | 550 | 0.744 | 0.08 | 0.84 | 0.8857 | 0.8087 |
| k_idioms (ours) | 100 | 0.88 | 0.44 | 0.72 | 0.6667 | 0.7586 |
| xstest | 450 | 0.71 | 0.008 | 0.8667 | 0.9861 | 0.8256 |

## 2. Refusal behavior — WildGuard vs baseline

| Metric | Group | WildGuard-ko | Baseline |
|---|---|---|---|
| safe false-refusal ↓ | overall | 0.0267 | 0.0167 |
| safe false-refusal ↓ | k_idioms (ours) | 0.1 | 0.06 |
| safe false-refusal ↓ | xstest | 0.012 | 0.008 |
| unsafe refusal / recall ↑ | overall | 0.448 | 0.424 |
| unsafe refusal / recall ↑ | k_idioms (ours) | 0.34 | 0.32 |
| unsafe refusal / recall ↑ | xstest | 0.475 | 0.45 |
| refusal precision ↑ | overall | 0.9333 | 0.955 |
| refusal precision ↑ | k_idioms (ours) | 0.7727 | 0.8421 |
| refusal precision ↑ | xstest | 0.9694 | 0.9783 |
| refusal F1 ↑ | overall | 0.6054 | 0.5873 |
| refusal F1 ↑ | k_idioms (ours) | 0.4722 | 0.4638 |
| refusal F1 ↑ | xstest | 0.6376 | 0.6164 |
| behavior_acc ↑ (see note) | overall | 0.7345 | 0.7182 |
| behavior_acc ↑ (see note) | k_idioms (ours) | 0.62 | 0.6 |
| behavior_acc ↑ (see note) | xstest | 0.76 | 0.7444 |

## 3. Harmful-response rate (WildGuard-only)

| Group | safe queries | unsafe queries | unparsed-refusal |
|---|---|---|---|
| overall | 0.03 | 0.264 | 0 |
| k_idioms (ours) | 0.14 | 0.44 | 0 |
| xstest | 0.008 | 0.22 | 0 |

## 4. Per-example refusal agreement (WildGuard vs baseline)

- Compared: **550**  |  agreement: **0.9364**  |  Cohen's κ: **0.8083**

| | baseline: refusal | baseline: non-refusal |
|---|---|---|
| **WildGuard: refusal** | 98 | 22 |
| **WildGuard: non-refusal** | 13 | 417 |
