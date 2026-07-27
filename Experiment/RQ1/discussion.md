# RQ1 — 멀티모달 입력의 영향성

**설정**: Qwen3.5-27B, ID 분포, full 프롬프트, n=100 seeds. B(VLM+GCRL) 성능을 입력 모달리티별 비교.

## 결과 (safe_success / success / collision)
| modality | safe_success | success | collision |
|---|---|---|---|
| RGB+Height | 0.810 | 0.810 | 0.010 |
| RGB only | 0.840 | 0.850 | 0.010 |
| Height only | 0.820 | 0.820 | 0.000 |

## 고찰
- 지배 모달리티 = **RGB** (RGB only 0.850 vs Height only 0.820). 단일 입력 성능 격차가 큼.
- **결합 효과**: RGB+Height 0.810 vs 최고 단일 RGB 0.850 → **간섭적: 부정확한 모달리티를 더하면 오히려 하락**.
- **단계 분해**(핵심): pile 도달 RGB 1.00 vs Height 0.86 → 물체 위치 파악은 **RGB** 우위. / truck 도달(장애물 회피) both 0.82 vs RGB 0.85 → 회피엔 **height map** 기여. 두 모달리티가 서로 다른 하위과제에 강점.
- 충돌률: RGB only 0.010, Height only 0.000, both 0.010.
