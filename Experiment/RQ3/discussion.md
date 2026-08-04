# RQ3 — OOD에서 VLM 서브골의 대처 효과

**설정**: Qwen3.5-35B-A3B, full 프롬프트, both 모달리티, n=100 seeds. ID vs OOD(더 높고 큰 장애물, 분포 밖)에서
A(단독 GCRL 직접)와 B(VLM+GCRL) 비교.

## 결과 (safe_success)
| dist | A (GCRL only) | B (VLM+GCRL) | Δ(B−A) |
|---|---|---|---|
| ID | 0.000 | 0.420 | +0.420 |
| OOD | 0.000 | 0.330 | +0.330 |

## 고찰
- ID에서 VLM 이득 Δ=+0.420, OOD에서 Δ=+0.330.
- OOD 이득이 작거나 유사 → OOD에서 VLM 계획 품질도 함께 저하.
- 단독 GCRL은 OOD에서 0.000→0.000로 유지, VLM 가이드가 이를 부분 회복.
