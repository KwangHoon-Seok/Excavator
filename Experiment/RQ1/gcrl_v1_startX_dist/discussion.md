# RQ1 — 멀티모달 입력의 영향성  (GCRL 컨트롤러: gcrl_v1_startX_dist, 3M step)

**설정**: Qwen3.5-27B, ID 분포, full 프롬프트, n=100 seeds. B(VLM+GCRL) 성능을 입력 모달리티별 비교.

## 결과 (safe_success / success / collision)
| modality | safe_success | success | collision |
|---|---|---|---|
| RGB+Height | 0.410 | 0.680 | 0.300 |
| RGB only | 0.570 | 0.920 | 0.350 |
| Height only | 0.576 | 0.717 | 0.182 |

## 고찰
- 지배 모달리티 = **RGB** (RGB only 0.920 vs Height only 0.717). 단일 입력 성능 격차가 큼.
- **결합 효과**: RGB+Height 0.680 vs 최고 단일 RGB 0.920 → **간섭적: 부정확한 모달리티를 더하면 오히려 하락**.
- **단계 분해**(핵심): pile 도달 RGB 1.00 vs Height 0.76 → 물체 위치 파악은 **RGB** 우위. / truck 도달(장애물 회피) both 0.73 vs RGB 0.92 → 회피엔 **height map** 기여. 두 모달리티가 서로 다른 하위과제에 강점.
- 충돌률: RGB only 0.350, Height only 0.182, both 0.300.
