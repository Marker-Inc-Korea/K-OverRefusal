# Inter-judge agreement — gpt-oss-safeguard-120b vs gpt-5.5

- Label scheme: 3-class {refusal, deflection, compliance} on `refusal_clf_label`
- Models compared: EXAONE-4.0-32B, HyperCLOVAX-SEED-Text-Instruct-1.5B, gemini-3.5-flash, gpt-5.5, kanana-2-30b-a3b-instruct-2601, solar-pro-preview-instruct

## Overall (all groups)

| Model | n | 3-class agree | 3-class κ | binary-refusal agree | binary-refusal κ |
|---|---|---|---|---|---|
| **ALL MODELS** | 3296 | 0.868 | 0.769 | 0.9663 | 0.9251 |
| EXAONE-4.0-32B | 548 | 0.8431 | 0.7203 | 0.9471 | 0.8758 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 550 | 0.8855 | 0.8041 | 0.9891 | 0.9762 |
| gemini-3.5-flash | 550 | 0.9182 | 0.8287 | 0.9709 | 0.9298 |
| gpt-5.5 | 550 | 0.9164 | 0.8424 | 0.9582 | 0.9117 |
| kanana-2-30b-a3b-instruct-2601 | 548 | 0.8777 | 0.777 | 0.9599 | 0.9138 |
| solar-pro-preview-instruct | 550 | 0.7673 | 0.6537 | 0.9727 | 0.9388 |

## k_idioms (ours)

| Model | n | 3-class agree | 3-class κ | binary-refusal agree | binary-refusal κ |
|---|---|---|---|---|---|
| **ALL MODELS** | 600 | 0.8233 | 0.7137 | 0.97 | 0.9397 |
| EXAONE-4.0-32B | 100 | 0.74 | 0.6032 | 0.95 | 0.8964 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 100 | 0.87 | 0.7733 | 1.0 | 1.0 |
| gemini-3.5-flash | 100 | 0.89 | 0.7771 | 0.97 | 0.9231 |
| gpt-5.5 | 100 | 0.86 | 0.752 | 0.93 | 0.8603 |
| kanana-2-30b-a3b-instruct-2601 | 100 | 0.85 | 0.7449 | 0.98 | 0.96 |
| solar-pro-preview-instruct | 100 | 0.73 | 0.5787 | 0.99 | 0.9799 |

## xstest

| Model | n | 3-class agree | 3-class κ | binary-refusal agree | binary-refusal κ |
|---|---|---|---|---|---|
| **ALL MODELS** | 2696 | 0.878 | 0.7782 | 0.9655 | 0.9198 |
| EXAONE-4.0-32B | 448 | 0.8661 | 0.7467 | 0.9464 | 0.8688 |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | 450 | 0.8889 | 0.7996 | 0.9867 | 0.9686 |
| gemini-3.5-flash | 450 | 0.9244 | 0.8407 | 0.9711 | 0.9311 |
| gpt-5.5 | 450 | 0.9289 | 0.8621 | 0.9644 | 0.9229 |
| kanana-2-30b-a3b-instruct-2601 | 448 | 0.8839 | 0.7791 | 0.9554 | 0.9002 |
| solar-pro-preview-instruct | 450 | 0.7756 | 0.6592 | 0.9689 | 0.9246 |

## Overall 3x3 confusion (rows = gpt-oss-safeguard-120b, cols = gpt-5.5)

| A \ B | refusal | deflection | compliance |
|---|---|---|---|
| **refusal** | 1068 | 33 | 9 |
| **deflection** | 35 | 154 | 13 |
| **compliance** | 34 | 311 | 1639 |
