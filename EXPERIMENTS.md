# 실험 계획서 — VLM Planner + GCRL Controller (IEEE Access)

> 논문 실험의 **단일 소스(living doc)**. Metric 정의·실험 매트릭스·통계 프로토콜·구현 산출물을 고정한다.
> 설계 근거는 [RL_FRAMEWORK.md](RL_FRAMEWORK.md)·[Model/](Model/)·[VLM/](VLM/)·[Evaluate/](Evaluate/).

---

## 0. 논문 주장(Claim) 확정

계층 구조: **VLM planner(상위, 의미론적 지각·계획) + GCRL controller(하위, waypoint 추종)**.
VLM은 장애물을 의미론적으로 인식해 회피 subgoal을 만들고, GCRL은 그 subgoal을 추종하며 자체 충돌(지면·관절)만 신경 쓴다.

### Contribution
| | 내용 | 우선순위 |
|---|---|---|
| **C1** | **사전학습(pre-trained, training-free) VLM을 임무 플래너로 채택하는 것이 타당하다** | ★ 핵심 |
| **C2** | 모듈러 통합(planner를 교체해도 controller 재학습 불필요) | 보조 |
| **C3** | adaptive target entropy로 커리큘럼 α 붕괴 완화 | 보조 |

### C1의 정확한 서사 (중요)
- **주장한다**: off-the-shelf VLM은 태스크별 파인튜닝·특권정보(GT)·수동규칙 **없이도**, 손수 설계한 플래너에 **필적**하고 naive 베이스라인을 **크게 상회**하며, **하나의 모델로 언어지시에 따라 다중 태스크**를 계획한다 → 플래닝 계층으로 VLM 채택은 정당하다.
- **주장하지 않는다**: "VLM이 FSM보다 OOD에 강하다." VLM은 장애물 데이터로 학습하지 않았으므로 OOD 우월성은 근거가 없다. 씬 다양성 테스트는 "플래너가 씬 변화를 다뤄야 하니 당연히 평가"하는 것이지 robustness 우월 주장이 아니다.
- 헤드라인 표현: **"training-free 범용·언어조건 VLM 플래너 채택의 타당성"**. abstract/intro에서 OOD 언어는 배제.

### 확정된 설계 결정
1. **B2 = GT 특권정보 + 손튜닝 FSM** (B2에 최대한 유리하게). → "VLM은 GT·학습·규칙 없이 대등"을 최강으로 만든다.
2. **언어유연성(지시문 변경) 실험을 주요 실험으로 승격** (부록 데모 아님). pre-trained VLM 채택의 가장 강한 정당화 축.
3. GCRL 재학습 **3 시드** (mean±std).

---

## 1. 실험 계층 (Contribution ↔ 실험)

| 실험 | 검증 대상 | Contribution | 핵심 질문 |
|---|---|---|---|
| **E1** 하위제어기 능력 | GCRL이 공정한 "고정 기판"인가 | C2 전제 | reachable waypoint를 전 작업공간에서 자기충돌 없이 안정 추종? |
| **E2** 플래너 교체 통제실험 | VLM 채택 타당성 | **C1(핵심)** | 필요성·대등성·유연성 (아래 3 하위주장) |
| **E3** adaptive entropy ablation | 커리큘럼 α 붕괴 완화 | C3 | 표본효율·최종성공·엔트로피 붕괴? |
| **E4** 설계선택 ablation | goal-norm/random-start/reward 분해 | design choice(not contrib) | 각 요소 기여분 |

E2가 핵심. E1은 그 통제실험의 **내적 타당성 방어막**(제어기가 흔들리면 C1 결과 오염).

---

## 2. 평가 Metric 확정 (fix)

이제부터 모든 실험이 이 정의를 쓴다. 세 층위.

### (A) Mission-level — E2(C1) 주력
1 에피소드 = 한 플랜(subgoal 시퀀스)을 한 씬에서 **리셋 없이 순차 실행**([Evaluate/sequential.py](Evaluate/sequential.py) `execute` 방식).

| 지표 | 정의 | 방향 |
|---|---|---|
| **MSR** (Mission Success Rate) | **모든 subgoal 도달 AND 장애물 충돌 0회** | ★주지표, ↑ |
| **CR** (Collision Rate) | 궤적 중 팔–장애물 관통(min clearance < 0) 발생 에피소드 비율 | ↓ |
| **MC** (Min Clearance) | 궤적 전체 min dist(arm/bucket geoms ↔ obstacle/truck geoms) [m], mean & 5th-pct | ↑ |
| **SRR** (Subgoal Reach Rate) | 도달 subgoal / 전체 subgoal | ↑ |
| **PL** (Path Length) | bucket_tip 누적 이동거리 [m] — 우회 비용/효율 | ↓(과우회 감지) |
| **TTC** (Time-to-Complete) | 완료까지 제어 스텝(타임아웃 상한) | ↓ |
| **PV** (Plan Validity) | reachable envelope 안에 든 subgoal 비율(`env._is_reachable`) — 플래너 단독 | ↑ |

**충돌/clearance 계산**: 궤적 매 스텝 `mujoco.mj_geomDistance(arm/bucket geoms ↔ obstacle/truck geoms)`. [Model/excavator_env.py:91](Model/excavator_env.py#L91) `_self_collision` 패턴 재사용. `min<0`=충돌(CR), `min<0.2m`=unsafe.
→ MSR을 "충돌 없이"로 정의해야 B1(직선)이 goal에 도달해도 충돌로 MSR이 무너져 **장애물이 문제임**이 드러난다.

### (B) VLM-specific — E2 (C1 재프레이밍용 추가)
| 지표 | 정의 | 목적 |
|---|---|---|
| **SC** (Semantic Correctness) | plan.json `objects`의 타입·위치가 씬 GT와 일치하는 비율 | VLM이 씬을 옳게 지각하나 |
| **IA** (Instruction Adherence) | 지시한 태스크 달성(dig site/truck bed 도달) 비율 | 언어조건 유연성 |
| **공학비용표**(정성) | GT필요?·태스크학습?·손튜닝 규칙 LOC·지시조건화·신규태스크 추가비용 | B1/B2/B3 대조 → "절약분" |

### (C) Controller-level — E1 (대부분 구현됨)
[Model/callbacks.py](Model/callbacks.py) `FullTaskEvalCallback`이 이미 로깅:
- `eval_fulltask/success_rate`, region별 성공률(front/side/wide × near/mid/far × lowz/highz), `steps_to_success`, per-joint action saturation, `failures.csv`.

### (D) C3-level — E3 (이미 로깅)
- 표본효율: full-task success 0.5/0.8 도달 스텝, 성공곡선 AUC.
- 최종 성공률: `fulltask.csv`.
- 엔트로피 붕괴: `policy/entropy_actual` − `policy/target_entropy` gap, α 궤적([Model/callbacks.py:211](Model/callbacks.py#L211)).

---

## 3. E2 — VLM 채택 타당성 (C1 핵심)

### 3.1 플래너 4종 (GCRL·씬·metric 전부 고정, waypoint 생성기만 교체 = C2 통제)
| ID | 플래너 | 입력 | 역할 |
|---|---|---|---|
| **B0** Oracle | GT 장애물 → 최적 up-over-down waypoint | 특권정보 | 상한(upper bound) |
| **B1** Straight | task goal 1점만(중간 waypoint 없음) | 없음 | 하한 — "장애물이 문제다" 대조 |
| **B2** FSM/manual | **GT 장애물 + 손튜닝 규칙/템플릿** | 특권정보 | 손설계 대안(공학비용 대조) |
| **B3** VLM | RGB + 극좌표 gridmap(지각) | 이미지 | 제안기법([VLM/infer.py](VLM/infer.py)) |

### 3.2 3개 하위주장
| 하위주장 | 비교 | 증명 |
|---|---|---|
| **E2a 필요성** | **B3 ≫ B1** | 플래너가 진짜 의미론적 회피 수행(MSR↑·CR↓), trivial 아님 |
| **E2b 대등성** | **B3 ≈ B2(GT+FSM) → B0** | GT·학습·규칙 **없이** 손설계 수준 달성. 공학비용표로 절약분 정량화 |
| **E2c 유연성+한계** | 같은 씬 × **지시문 변경**(dig/load/avoid) | 언어조건 하나로 다중 태스크(FSM은 재코딩). + off-the-shelf 실패모드 정직 분석 |

### 3.3 씬 세팅 (용어: canonical vs held-out configurations — "OOD 붕괴"는 헤드라인 아님)
- **canonical**: [scene.py](scene.py) 정규 레이아웃 ± 소폭 지터(장애물 xy ±0.2m, 높이 ±0.1m). B2를 여기서 튜닝.
- **held-out**: 축별 체계 변형 — 장애물 개수 0/1/2/3, 방위 ±90° 전개, 높이 0.5–3.5m, 구덩이 위치/유무, 트럭 좌/우, novel 배치(정면 장애물·이중 코리도·초고 장애물).
- breakdown(개수/방위/높이)은 **VLM 실패모드 국소화(E2c)**용.

### 3.4 기대 결과표 (논문 핵심 그림)
| | canonical MSR | held-out MSR | CR | 비고 |
|---|---|---|---|---|
| B0 Oracle | ~0.95 | ~0.90 | low | 상한 |
| B1 Straight | 낮음 | 낮음 | 높음 | 하한(의미론 필요성) |
| B2 FSM(GT) | 높음 | 규칙 일반성에 의존 | low | 손설계, 특권정보 |
| B3 VLM | **≈B2, →B0** | **≈B2** | low | 무학습·무GT·언어조건 |

해석: B2가 잘 버티면 → "동일 성능·공학비용 0·언어유연성"(비용 서사). B2가 재튜닝 없이 흔들리면 → "무학습으로 씬 다양성 커버"(단, per-scene 공학 불요로 프레이밍). **어느 결과든 C1 성립.**

---

## 4. E1 — 하위제어기 능력 (기판 고정)
- [Model/checkpoints/](Model/checkpoints/) 최종 config로 **3 시드** 재학습.
- 지표: 전체/region별 성공률, mean final dist, steps-to-success.
- 목표: 전 구역(특히 far/wide/highz) 도달 + 자기충돌 ~0 → C1 통제실험의 confound 제거.
- 기존 러너: [Evaluate/evaluate.py](Evaluate/evaluate.py), 기존 곡선 [Evaluate/plots/](Evaluate/plots/).

---

## 5. E3 — Adaptive Target Entropy (C3)
- 비교: fixed target_entropy vs adaptive(`curr_te_boost`, [Model/callbacks.py](Model/callbacks.py) `CurriculumCallback`), **각 3 시드**.
- 지표: 표본효율(0.5/0.8 도달 스텝), 최종 full-task 성공, 엔트로피 붕괴 곡선, AUC.
- 기존 로그: [Model/logs/Version1/](Model/logs/) `target_entropy`, `target_entropy_goal_normalization`, `init_alpha`.

---

## 6. E4 — 설계선택 Ablation (design choice, not contribution)
- goal normalization on/off, random_start on/off, reward 분해(sparse / sparse_dense / reward_decomposed).
- 기존 로그: [Model/logs/Version2/](Model/logs/) `without_normalize_goal_0706`, `without_random_position_0706`, `Total_reward_0705`.
- 서술: contribution 아님, "왜 이 설계인가"를 뒷받침하는 design justification.

---

## 7. 통계 프로토콜 (IEEE Access 대비)
- **GCRL 3 시드** — 제어기가 공유 기판이므로 시드가 전 조건에 전파.
- **씬 N=100/설정**, 고정 시드 생성 → 모든 플래너가 **동일 씬**을 품(paired).
- rate는 **Wilson CI**, B2 vs B3 MSR은 동일 씬 paired → **McNemar 검정**.
- VLM: greedy 디코딩(재현성) + 강건성용 3회 샘플 분산 부기.
- 보고: mean ± std across seeds, per-scene paired diff.

---

## 8. 구현 산출물 (우선순위)

| # | 파일 | 내용 | 상태 |
|---|---|---|---|
| 1 | `Evaluate/collision.py` | arm/bucket ↔ obstacle/truck geom clearance 검출(궤적 min·충돌 플래그). `_self_collision` 재사용 | ⬜ **최우선(없으면 C1 측정 불가)** |
| 2 | `Evaluate/scene_gen.py` | canonical/held-out 씬 생성기([scene.py](scene.py) 파라미터화: 개수/xy/높이/구덩이/트럭, 시드 고정) | ⬜ |
| 3 | `Evaluate/planners/` | B0 oracle · B1 straight · B2 fsm(GT+규칙) · B3 vlm 래퍼. 공통 IF `plan(scene, instruction) -> [subgoals]` | ⬜ |
| 4 | `Evaluate/run_mission.py` | (planner × scene-set) → execute + collision → **metric CSV** | ⬜ E2 메인 러너 |
| 5 | `FullTaskEvalCallback` self-collision rate | E1 지표 1줄 추가 | ⬜ |

**metric CSV 스키마(고정)**: `scene_id, planner, seed, instruction, mission_success, collision, min_clearance, subgoal_reach_rate, path_length, ttc, plan_validity, semantic_correctness, instruction_adherence`

---

## 9. 실행 순서
1. `collision.py` + `scene_gen.py` + `run_mission.py` 스캐폴드 → **B1/Oracle로 파이프라인 관통 검증**(VLM은 나중).
2. GCRL 3시드 재학습 → E1 표 확정(기판 고정).
3. B2 FSM을 canonical에서 튜닝 → canonical/held-out 전체 실행 → E2 메인표.
4. E2c 지시문 변경 실험(언어유연성).
5. E3·E4는 로그 축적 중 → 시드만 3으로 보강.

---

## 부록 A. 미해결/합의 필요
- B2 FSM 규칙 상세(임계값·템플릿 형태) — canonical 튜닝 시 확정.
- held-out 씬 축별 샘플 수 배분(개수/방위/높이 격자).
- SC(semantic correctness) 매칭 기준(위치 허용오차, 타입 confusion matrix).
</content>
</invoke>
