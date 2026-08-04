# RQ1 — 멀티모달 (pile/truck 위치 랜덤화, Cartesian, v1)

**설정**: Qwen3.5-27B, ID 분포, full 프롬프트, n=100 seeds. B(VLM+GCRL) 성능을 입력 모달리티별 비교.

## 결과 (safe_success / success / collision)
| modality | safe_success | success | collision |
|---|---|---|---|
| RGB+Height | 0.720 | 0.980 | 0.280 |
| RGB only | 0.060 | 0.060 | 0.100 |
| Height only | 0.610 | 0.990 | 0.380 |

## 고찰
- 지배 모달리티 = **Height map** (RGB only 0.060 vs Height only 0.990). 단일 입력 성능 격차가 큼.
- **결합 효과**: RGB+Height 0.980 vs 최고 단일 Height 0.990 → **간섭적: 부정확한 모달리티를 더하면 오히려 하락**.
- **단계 분해**(핵심): pile 도달 RGB 0.29 vs Height 0.99 → 물체 위치 파악은 **RGB** 우위. / truck 도달(장애물 회피) both 0.98 vs RGB 0.30 → 회피엔 **height map** 기여. 두 모달리티가 서로 다른 하위과제에 강점.
- 충돌률: RGB only 0.100, Height only 0.380, both 0.280.
