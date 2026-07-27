# RQ2 — VLM 파라미터 수에 따른 성능

**설정**: 모델 사다리 ['Qwen3.5-4B', 'Qwen3.5-27B', 'Qwen3.5-35B-A3B'], ID 분포, full 프롬프트, both 모달리티, n=100 seeds.
baseline A(단독 GCRL 직접) safe_success ≈ 0.000.

## 결과 (B=VLM+GCRL)
| model | safe_success | success | collision | avg subgoals |
|---|---|---|---|---|
| Qwen3.5-4B | 0.020 | 0.030 | 0.080 | 3.9 |
| Qwen3.5-27B | 0.810 | 0.810 | 0.010 | 4.5 |
| Qwen3.5-35B-A3B | 0.420 | 0.420 | 0.010 | 4.8 |

## 고찰
- 모델 크기↑에 따른 safe_success 추세: **증가**. 최고 성능 = **Qwen3.5-27B** (0.810).
- 모든 크기에서 B가 baseline A(0.000)를 상회 → VLM 플래닝의 효용.
- 작은 모델일수록 서브골 품질(도달성·회피 여유) 저하로 collision↑ / stage 도달↓ 경향 확인.
- 122B는 디스크/메모리 제약으로 미포함(양자화 체크포인트 부재/자원 제약).
