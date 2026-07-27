# RQ1 — 멀티모달 입력의 영향성

**설정**: Qwen3.5-27B, ID 분포, full 프롬프트, n=100 seeds. B(VLM+GCRL) 성능을 입력 모달리티별 비교.

## 결과 (safe_success / success / collision)
| modality | safe_success | success | collision |
|---|---|---|---|
| RGB+Height | 0.360 | 0.650 | 0.420 |
| RGB only | 0.090 | 0.090 | 0.080 |
| Height only | 0.310 | 0.560 | 0.400 |

## 고찰
- 지배 모달리티 = **Height map** (RGB only 0.090 vs Height only 0.560). 단일 입력 성능 격차가 큼.
- **결합 효과**: RGB+Height 0.650 vs 최고 단일 Height 0.560 → 상보적(결합이 최고 단일 초과).
- **단계 분해**(핵심): pile 도달 RGB 0.17 vs Height 0.57 → 물체 위치 파악은 **RGB** 우위. / truck 도달(장애물 회피) both 0.97 vs RGB 0.43 → 회피엔 **height map** 기여. 두 모달리티가 서로 다른 하위과제에 강점.
- 충돌률: RGB only 0.080, Height only 0.400, both 0.420.
