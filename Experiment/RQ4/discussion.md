# RQ4 — 프롬프트 엔지니어링의 영향

**설정**: Qwen3.5-35B-A3B, ID, both 모달리티, n=100 seeds. full 프롬프트에서 요소를 하나씩 제거.

## 결과 (safe_success / collision)
| prompt | safe_success | collision |
|---|---|---|
| full | 0.420 | 0.010 |
| no_kinematics | 0.330 | 0.010 |
| no_cot | 0.140 | 0.110 |
| no_clearance | 0.350 | 0.060 |
| minimal | 0.000 | 0.020 |

## 고찰
- full 대비 safe_success 하락폭: no_kinematics +0.090, no_cot +0.280, no_clearance +0.070, minimal +0.420.
- 가장 큰 영향 요소 = **minimal** (제거 시 -0.420) → 이 요소가 전체 시스템 성능에 핵심.
- minimal(좌표+태스크+출력형식만)은 full 대비 크게 하락 → 프롬프트 스캐폴딩(운동학/CoT/clearance)이 중요.
