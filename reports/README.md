# Reports

주제별 실험 리포트(목적·데이터·방법·결과·해석·한계). 원시 로그는 포함하지 않고, 각 리포트의 `results/`에 처리된 결과 요약(csv/json)만 둔다.

| 리포트 | 상태 | 한 줄 요약 |
|---|:--:|---|
| [orbench](orbench/리포트.md) | 완료 | OR-Bench(KO) 도입. gpt 번역은 거절 폭증(214/1319)으로 실패 → instructTrans 번역 + gpt 필터 2단계로 1312/1319 확보. over-refusal metric 추가, XSTest 호환 유지 |
| [merged](merged/리포트.md) | 완료 | k_idioms+XSTest+OR-Bench+OKTest+PHTest 통합셋(435)으로 10개 로컬 모델 평가. 과잉거부–안전성 트레이드오프 확인(EEVE 정상응답 0.90/유해거부 0.40, A.X behavior_acc 0.67 최상위), k_idioms가 XSTest보다 어려움 |
