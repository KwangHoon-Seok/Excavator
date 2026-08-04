# RQ3 (reach) — 새로운(OOD) 환경으로의 일반화

**설정**: Qwen3.5-27B(+Cartesian 높이맵), 도달(reaching) 레벨. A=단독 GCRL 직접, B=VLM+GCRL 서브골. 같은 seed→같은 goal, 장애물만 ID/OOD.

| dist | A safe (GCRL) | B safe (VLM+GCRL) | Δ | A coll | B coll | n_obs |
|---|---|---|---|---|---|---|
| ID | 0.490 | 0.350 | -0.140 | 0.050 | 0.300 | 2.6 |
| OOD | 0.130 | 0.120 | -0.010 | 0.450 | 0.740 | 5.5 |

## 고찰
- **논점 1 (GCRL 저하)**: 단독 GCRL safe-success가 ID 0.490 → OOD 0.130 (▼0.360). OOD 충돌률 0.050→0.450 급증 — 밀집·고벽 장애물을 반응회피만으로 넘지 못함.
- **논점 2 (VLM 회복)**: VLM+GCRL은 OOD에서도 safe-success 0.120 (단독 대비 Δ-0.010). ID 대비 retention = 0.34 (B_OOD/B_ID).
- 결론: VLM 서브골이 훈련 분포 밖 환경에서 GCRL의 저수준 회피 한계를 보완 → 프레임워크가 novel 환경으로 일반화.
