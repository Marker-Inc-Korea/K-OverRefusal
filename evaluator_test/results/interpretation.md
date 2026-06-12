# Evaluator-test 결과 해석 (WildGuard-ko 비교)

- 테스트 evaluator: `iknow-lab/llama-3.2-3B-wildguard-ko-2410`
- 기존 judge (baseline): `openai/gpt-oss-safeguard-120b` (3-class refusal/deflection/compliance)
- gold label: 기존 + 직접 작성한 safe / unsafe
- k_idioms = 직접 제작한 한국어 관용어 pseudo-harmful 세트 (xstest와 분리 보고)
- 대상: 11개 모델, 그룹별 집계

> **목적**: 자동 평가기(WildGuard-ko)가 (1) 내가 작성한 gold label, (2) 기존 judge와
> 각각 얼마나 일치/상이한지를 **비교·진단**하는 것. 평가기 교체가 목적이 아니다.

---

## 한눈에 보는 결론

| 비교 축 | 일치 정도 | 요약 |
|---|---|---|
| WildGuard ↔ 기존 judge(safeguard) — refusal 행동 | 일치율 ~0.95, κ ~0.88 | 두 judge의 거절 판정은 거의 동일 |
| WildGuard ↔ 내 gold label — xstest | acc 0.87 | 잘 맞음, safe 오탐 거의 0 |
| WildGuard ↔ 내 gold label — **k_idioms (직접 제작)** | acc 0.72 | **크게 어긋남 — safe 관용어 오탐 ~45%** |

핵심: 자동 평가기가 내 라벨과 가장 크게 갈리는 지점은 **직접 제작한 k_idioms**.
한국어 관용 표현(예: "혈을 뚫다", "씹어먹다")을 글자 그대로 위험하다고 읽어내는 경향이 강함.

---

## 1. 테스트 evaluator vs 기존 judge (WildGuard vs gpt-oss-safeguard)

두 judge가 응답을 "거절 / 비거절"로 보는 판단은 거의 일치한다
(11개 모델 평균 refusal agreement **0.949**, Cohen's κ **0.884**).

| 그룹 | false-refusal (WG / BL) | recall (WG / BL) | refusal-F1 (WG / BL) |
|---|---|---|---|
| overall  | 0.097 / 0.082 | 0.692 / 0.658 | 0.760 / 0.744 |
| k_idioms | 0.329 / 0.295 | 0.726 / 0.706 | 0.702 / 0.699 |
| xstest   | 0.051 / 0.039 | 0.684 / 0.646 | 0.778 / 0.758 |

→ WildGuard가 safeguard보다 아주 약간 더 거절-민감(false-refusal +0.015, recall +0.03)하지만
두 judge의 경향성은 본질적으로 동일. 즉 어느 judge를 보든 **모델 간 상대적 경향/순위는 일관**.

## 2. 테스트 evaluator vs 작성한 label (harmful_request 판정)

WildGuard의 `Harmful request: yes/no`가 기존 label 얼마나 맞는가:

| 그룹 | accuracy | unsafe→flagged (recall) | safe→flagged (오탐) |
|---|---|---|---|
| overall  | 0.84 | 0.75  | 0.082 |
| **k_idioms (직접 제작)** | **0.72** | 0.885 | **0.447** |
| xstest   | 0.87 | 0.716 | **0.009** |

- **xstest**: 거의 완벽한 보정. 안전한 프롬프트를 harmful로 오탐하는 비율이 0.9%뿐.
  다만 *unsafe는 약 28%를 놓침(보수적)*.
- **k_idioms**: 안전한 관용어 프롬프트의 **44.7%를 harmful request로 오탐**(safe 50개 중 ~22개).
  accuracy도 0.72로 급락.


## 3. 응답 자체의 유해성 (harmful_response, WildGuard 기준)

응답이 실제로 유해하다고 판정된 비율은 매우 낮음 — unsafe 쿼리 평균 **5.5%**, safe 쿼리 **1.6%**.
즉 모델들이 거절/우회는 해도 유해 콘텐츠를 거의 내놓지 않음.

---

## 시사점 — 새로운 moderation tool 개발 동기

- 기성 한국어 안전 평가기(WildGuard-ko)는 **한국어 관용어(k_idioms)에서 safe를 harmful로 오판하는 비율이 높음**.
- 따라서 한국어 관용어·비유의 양성(benign) 의도를 구분하는 데 초점을 둔
  **전용 moderation tool 개발 시, 핵심 평가 축은 k_idioms의 safe 오탐율(false-positive) 감소**.
