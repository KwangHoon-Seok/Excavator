# RQ3 — Does the framework generalize to novel (OOD) environments?

두 레벨의 상보적 증거로 답한다. **(A) reaching** 실험은 저수준 GCRL 컨트롤러가 분포 밖 장애물 환경에서
**저하**함을 격리해 보이고(논점1), **(B) loading mission** 실험은 VLM+GCRL 프레임워크가
그 환경에서도 **작동·일반화**함을 보인다(논점2). 컨트롤러 체크포인트는 두 실험 모두 `v4base.zip`로 고정.

## (A) 저수준 GCRL의 OOD 저하 — reaching, n=100/분포
같은 seed→같은 목표, 장애물만 ID(≤3개·0.7–2.5 m, 훈련분포) vs OOD(5–7개·2.5–3.5 m 벽, 분포 밖).
VLM = Qwen3.5-27B(+Cartesian 높이맵).

| dist | GCRL-only safe | GCRL-only collision | n_obs |
|---|---|---|---|
| ID  | **0.49** | 0.05 | 2.6 |
| OOD | **0.13** | 0.45 | 5.5 |

→ 단독 GCRL의 안전 성공률이 ID 0.49 → OOD 0.13으로 급락하고, 충돌률이 0.05 → 0.45로 급증.
훈련 분포 밖의 밀집·고벽 장애물을 저수준 반응 회피만으로는 넘지 못한다. **(논점1 입증)**

## (B) VLM+GCRL 프레임워크의 OOD 일반화 — loading mission, n=100/분포
흙더미→트럭 상차 미션. A=단독 GCRL 직접, B=VLM+GCRL 서브골. ID vs OOD(더 높고 큰 장애물).

| dist | GCRL-only (A) safe | VLM+GCRL (B) safe | B collision |
|---|---|---|---|
| ID  | 0.00 | **0.42** | 0.01 |
| OOD | 0.00 | **0.33** | 0.23 |

→ 단독 GCRL은 장애물 우회 계획이 없어 ID·OOD 모두 미션 자체를 완수하지 못한다(0.00).
VLM+GCRL은 ID 0.42, OOD 0.33의 안전 성공률을 달성하며 **retention = 0.33/0.42 ≈ 0.79**로
분포 밖 환경에서도 성능을 대부분 유지한다. **(논점2 입증)**

## 종합
저수준 정책만으로는 새로운 환경에서 성능이 무너지지만(A), VLM의 의미론적 인지와 서브골 계획을
결합한 프레임워크는 그 환경에서도 동작하며 일반화한다(B). → **프레임워크는 novel 환경으로 일반화한다.**

## 한계 및 향후 과제 (정직 서술)
- reaching 실험에서 **VLM 가이드가 OOD 안전 성공률을 단독 GCRL 이상으로 끌어올리지는 못했다**
  (Δsafe ≈ 0). 원인: 굴착기는 목표 도달을 위해 **캐빈을 스윙(회전)**하므로, 팔(붐·스틱·버킷)이
  장애물의 방위각 구간을 **쓸고 지나가며** 충돌한다. 버킷 tip의 (직교좌표) 경로만 안전하게 만드는
  계획·검증으로는 이 **스윙 스윕 충돌**을 제거하지 못한다(기하 검증기 적용 시 오히려 충돌↑ 확인:
  ID 0.24→0.30, OOD 0.61→0.74). 밀집 3.5 m 벽 OOD에서는 많은 (목표, 장애물) 조합에 대해
  충돌 없는 스윙 경로가 애초에 존재하지 않을 수 있다.
- mission에서 프레임워크가 일반화하는 이유는 트래버스 경로 기하가 스윙 통로를 남기기 때문으로 해석.
- **향후**: 극좌표(방위각 θ, 반경 r) 기반의 **스윙-인지 계획/검증**(장애물 방위 섹터 회피),
  또는 OOD 장애물 밀도·높이의 연속 스윕으로 프레임워크 일반화의 경계를 정량화.

## 산출물
- reaching per-seed: `Experiment/RQ3_reach/data_{id,ood}.csv`, 표 `table.tex`, 그림 `figure.png`
- mission per-seed: `Experiment/RQ3/data_{id,ood}.csv`, 표 `table.tex`, 그림 `figure.png`
- 결합 그림: `Experiment/RQ3_reach/rq3_combined_figure.png`
- 논문 서브섹션: `Experiment/RQ3_reach/rq3_subsection.tex`
- (부가) 검증기 ablation: `Experiment/RQ3_reach_val/` (스윙 스윕 한계 근거)
