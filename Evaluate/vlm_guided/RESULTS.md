# VLM-guided GCRL — A vs B paired 실험 결과

**질문**: VLM 서브골이 obstacle-aware GCRL 성능을 실제로 끌어올리는가?

**설정**: 학습 env(장애물 0.7~2.8m·goal-aware 배치), GCRL=v4base(SAC+HER, obstacle-aware).
A=goal 직접 / B=VLM 서브골 relay(마지막=goal). 동일 seed paired. VLM=claude CLI(haiku/sonnet/opus).
metric: safe_success(도달+무충돌) / success(도달) / collision(min_gap≤0).

---

## ★ 최종 결론: **YES (조건 명확)**
교란을 제거한 **진짜 회피 시나리오**에서 VLM 서브골은 GCRL 성능을 **명확히 끌어올린다**. baseline(A)이 전부 실패(safe=0)하는
과제에서 B는 safe_success를 0.35~0.47, 도달을 0.88~0.94까지 회복. **이득은 VLM 능력에 단조 증가**하며,
**궤적정형(smooth) 프롬프트로 증폭**된다(특히 약한 VLM). 이전의 "opus가 해롭다/비단조"는 **오염된 평가셋의 착시**였다.

---

## 1. 평가셋 오염 — 방법론적 핵심
raw seed의 대부분이 회피와 무관(free-space feasibility 스크린, 200 seed):
| 분류 | 비율 | |
|---|---|---|
| 저-z(z<1.0) | 33% | 버킷 바닥충돌 → GCRL 약점(회피 아님) |
| GOAL 불가(무장애물도 실패) | 26% | far+high 등 도달 자체 불가 |
| 장애물 안 막음(trivial) | 34% | 둘 다 성공 |
| **✅ 진짜 회피** | **8.5%** | z≥1.0 + free성공 + 장애물로 A가 실패/충돌 |

→ **오염셋(20 raw)에서는 opus가 최악(−0.10)으로 보였으나, 이는 trivial 케이스에서 opus의 적극적 계획이
불필요한 충돌을 추가한 산물.** 깨끗한 셋으로 재측정하면 결론이 뒤집힌다.

## 2. 깨끗한 17 seed 결과 (A: safe=0.000 · succ=0.471 · coll=0.471)
| VLM | 프롬프트 | 서브골수 | B_safe | B_succ | B_coll | safe승/17 |
|-----|---------|---------|--------|--------|--------|-----------|
| haiku  | baseline | 1.9 | 0.118 | 0.471 | 0.412 | 2 |
| haiku  | **smooth** | 6.7 | **0.412** | **0.882** | 0.471 | 7 |
| sonnet | baseline | 2.0 | 0.294 | 0.765 | 0.471 | 5 |
| sonnet | **smooth** | 7.4 | 0.353 | **0.941** | 0.588 | 6 |
| opus   | baseline | 2.7 | 0.412 | 0.882 | 0.471 | 7 |
| opus   | **smooth** | 6.9 | **0.471** | 0.882 | 0.412 | 8 |

**읽기**:
- **baseline 단조**: safe haiku 0.118 < sonnet 0.294 < opus 0.412. **더 좋은 VLM = 더 나은 회피.**
- **smooth 효과(diminishing returns)**: safe Δ = haiku **+0.294**, sonnet +0.059, opus +0.059.
  약한 모델일수록 궤적정형 프롬프트의 이득이 큼(스캐폴딩이 부족한 추론을 보완).
- **도달(success) 개선**: haiku 0.471→**0.882**, sonnet 0.765→**0.941**, opus 0.882(포화). 촘촘한 waypoint가 도달을 크게 향상.
- **부작용**: smooth의 조밀 waypoint는 장애물 근처 노출을 늘려 **충돌이 다소 증가할 수 있음**(sonnet coll 0.471→0.588).
- **최고**: opus+smooth = safe 0.471(8/17). 실용 통찰: **haiku+smooth(0.412) ≈ opus+baseline(0.412)** — 좋은 프롬프트가 VLM 능력을 대체.

## 3. VLM 장애물 지각 정확도 (plan.json vs GT)
| Model | recall | precision | 위치오차 | θ오차 | 높이오차 | 허위(FP) |
|---|---|---|---|---|---|---|
| haiku | 0.59 | 0.69 | 0.47m | 8.9° | 0.34m | 11(환각) |
| sonnet | 0.83 | 1.00 | 0.19m | 3.3° | 0.28m | 0 |
| opus | 0.85 | 1.00 | **0.14m** | **2.8°** | **0.19m** | 0 |

- 지각은 opus>sonnet>haiku 단조. **지각이 좋을수록 깨끗한 셋에서 계획도 좋음**(2와 일치) — 지각과 유용성이 정렬됨.
- (오염셋에선 이 정렬이 깨져 보였으나, 그건 평가셋 문제였음.)

## 4. 아키텍처 한계
GCRL은 obstacle-aware(clearance obs + TTC 페널티)이나 **soft**할 뿐, desired_goal은 반드시 추종 → **나쁜 서브골 거부권 없음**.
서브골이 안전 도달 불가능한 위치에 찍히면 팔이 장애물을 뚫고 감. 즉 **서브골 품질 = 단일 실패점**.
→ smooth 프롬프트는 VLM-side 완화. 추가로 **인터페이스 서브골 sanitization**(clearance 위반 서브골 투영/거부) 또는
**action shielding**(CBF/TTC veto)으로 GCRL-side recourse를 주면 강건해질 것.

## 한계 / 다음
- n=17(승패 1건=6%p) — 방향·메커니즘 일관하나 절대수치 노이즈. 유효 seed 확대(스크리너로 40~50개) 권장.
- 장애물 높이 0.7~2.8m: 저-높이는 수직통과 쉬움. 회피 난이도 통제 위해 높이/배치 세분화 여지.
- 산출물: `out_clean/`(baseline), `out_smooth/`(smooth) 각 `results_*.csv` + `seed*/rollout_{A,B}.gif`.
