# WildGuard-ko evaluator test — summary across models

- WildGuard evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- Baseline: `openai/gpt-oss-safeguard-120b`
- k_idioms = our hand-made split, reported independently.

## Overall (all groups)

| Model | harmful-req acc ↑ | WG false-ref ↓ | BL false-ref ↓ | WG recall ↑ | BL recall ↑ | WG refusal-F1 ↑ | BL refusal-F1 ↑ | refusal agree | κ |
|---|---|---|---|---|---|---|---|---|---|
| EXAONE-4.0-32B | 0.8345 | 0.0467 | 0.0567 | 0.656 | 0.5847 | 0.7663 | 0.7073 | 0.9215 | 0.8165 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 0.8491 | 0.15 | 0.12 | 0.672 | 0.648 | 0.7257 | 0.7232 | 0.9582 | 0.9107 |
| gemini-3.5-flash | 0.8364 | 0.01 | 0.0033 | 0.62 | 0.624 | 0.7598 | 0.7666 | 0.9727 | 0.9333 |
| gpt-5.5 | 0.8345 | 0.05 | 0.0567 | 0.788 | 0.772 | 0.8528 | 0.8391 | 0.9455 | 0.8847 |
| kanana-2-30b-a3b-instruct-2601 | 0.8655 | 0.06 | 0.05 | 0.78 | 0.7258 | 0.8423 | 0.8126 | 0.9599 | 0.914 |
| solar-pro-preview-instruct | 0.84 | 0.21 | 0.1467 | 0.652 | 0.576 | 0.6849 | 0.6575 | 0.9164 | 0.8227 |

## k_idioms (ours)

| Model | harmful-req acc ↑ | WG false-ref ↓ | BL false-ref ↓ | WG recall ↑ | BL recall ↑ | WG refusal-F1 ↑ | BL refusal-F1 ↑ | refusal agree | κ |
|---|---|---|---|---|---|---|---|---|---|
| EXAONE-4.0-32B | 0.75 | 0.2 | 0.18 | 0.7 | 0.6 | 0.7369 | 0.6742 | 0.9215 | 0.8165 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 0.71 | 0.48 | 0.4 | 0.78 | 0.76 | 0.6902 | 0.7037 | 0.9582 | 0.9107 |
| gemini-3.5-flash | 0.71 | 0.0 | 0.0 | 0.56 | 0.56 | 0.7179 | 0.7179 | 0.9727 | 0.9333 |
| gpt-5.5 | 0.72 | 0.08 | 0.16 | 0.84 | 0.88 | 0.875 | 0.8627 | 0.9455 | 0.8847 |
| kanana-2-30b-a3b-instruct-2601 | 0.81 | 0.22 | 0.2 | 0.84 | 0.8 | 0.8156 | 0.8 | 0.9599 | 0.914 |
| solar-pro-preview-instruct | 0.69 | 0.44 | 0.42 | 0.64 | 0.66 | 0.6154 | 0.6346 | 0.9164 | 0.8227 |

## xstest

| Model | harmful-req acc ↑ | WG false-ref ↓ | BL false-ref ↓ | WG recall ↑ | BL recall ↑ | WG refusal-F1 ↑ | BL refusal-F1 ↑ | refusal agree | κ |
|---|---|---|---|---|---|---|---|---|---|
| EXAONE-4.0-32B | 0.8533 | 0.016 | 0.032 | 0.645 | 0.5808 | 0.7748 | 0.7165 | 0.9215 | 0.8165 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 0.88 | 0.084 | 0.064 | 0.645 | 0.62 | 0.7371 | 0.7294 | 0.9582 | 0.9107 |
| gemini-3.5-flash | 0.8644 | 0.012 | 0.004 | 0.635 | 0.64 | 0.7697 | 0.7781 | 0.9727 | 0.9333 |
| gpt-5.5 | 0.86 | 0.044 | 0.036 | 0.775 | 0.745 | 0.847 | 0.8324 | 0.9455 | 0.8847 |
| kanana-2-30b-a3b-instruct-2601 | 0.8778 | 0.028 | 0.02 | 0.765 | 0.7071 | 0.85 | 0.8163 | 0.9599 | 0.914 |
| solar-pro-preview-instruct | 0.8733 | 0.164 | 0.092 | 0.655 | 0.555 | 0.7043 | 0.6647 | 0.9164 | 0.8227 |
