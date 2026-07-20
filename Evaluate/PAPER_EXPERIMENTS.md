# 논문 실험 계획 (Paper Experiments) — VLM-guided Obstacle-aware GCRL

VLM planner + GCRL controller 프레임워크의 정당성을 5개 RQ로 tight하게 입증한다.
모든 실험은 **동일한 고정 설정** 위에서 **단일 변인만** 바꿔 통제한다.

---

## 0. 고정 설정 (Fixed setup) — 전 실험 공통

| 요소 | 값 | 비고 |
|---|---|---|
| **GCRL (fixed)** | `Model/checkpoints/GCRL_fixed.zip` (=v4base) | SAC+HER, soft-TTC w=0.2, distance curriculum. **어려운 장애물 분포**(높이≤2.8·goal여유0.4·경로t0.75)로 ~1.34M 학습. 4-way bake-off(soft/constrained×dist/auto)에서 최고→fix |
| **VLM** | Claude CLI (haiku/sonnet/opus) | RGB+gridmap+prompt → sub-goal JSON. 로컬 Qwen 대체 |
| **Env** | MuJoCo obstacle-aware, obs 20차원 | `Model/Version4/config.yaml`(신 분포: 높이2.5·goal여유0.6·경로t0.6) 기준 |
| **주지표** | success(도달), **safe-success**(도달+무충돌), collision rate | `min_gap≤0`→충돌 |
| **보조지표** | 인지 정확도(recall/precision/위치·방위·높이 오차) | RQ1·RQ4 |

### 공통 평가 프로토콜 (엄격성 담보)
1. **Paired 비교**: 동일 seed → **동일 장애물·goal** → 변인만 차이 (교란 제거).
2. **Feasibility 스크리닝**: 회피 RQ(2,4,5)는 **"목표 도달 가능 + 장애물이 경로 차단"** 유효 시나리오만 선별(무작위의 ~91%는 회피 무관 — 저-z/도달불가/미차단). `Evaluate/vlm_guided/screen_valid` 방식.
3. **고정 seed set, n≥50** (통계 신뢰). VLM 계획은 **캐시+병렬**(claude CLI ~수분/call).
4. **정량(CSV) + 정성(rollout GIF)** 이중 검증.

---

## RQ1. VLM의 의미론적 인식이 필수적인가?
**가설**: 높이·기하 정보(gridmap)만으로는 **목표(흙더미)와 장애물을 구분 불가** → RGB 기반 의미 인식이 있어야 올바른 목표 선택·계획이 가능.

| 항목 | 설계 |
|---|---|
| **과업** | **상차(loading) 씬** — 흙더미(목표)+덤프트럭+랜덤 장애물. **목표를 좌표로 주지 않고** VLM이 지각으로 식별해야 함 (RQ2·4·5는 좌표 제공 → RQ1만 의미 식별 필요) |
| **변인(IV)** | 입력 modality: **Full(RGB+gridmap)** vs **No-RGB(gridmap only)** |
| **고정** | VLM 모델, GCRL, 프롬프트(§상차), 씬, seed |
| **비교** | Full → 흙더미=갈색(목표)/장애물=회색 박스 구분 O / No-RGB → 높이 blob만 보여 **목표=장애물 구분 X** |
| **지표** | ① **목표 식별 정확도**(pickup 대상이 흙더미인가) ② task success(올바른 목표 도달) ③ 오식별율(장애물을 목표로 잘못 선택) |
| **예상** | No-RGB는 흙더미를 목표로 인식 못 해 **오식별↑·success↓** |
| **필요/상태** | mission_scene(흙더미+트럭)·make_gridmap·system_prompt **보유** → **No-RGB 프롬프트(RGB 참조 제거) + 러너 신규**. seed별 씬 n≥30 |

**엄격성 포인트**: Full/No-RGB는 **동일 씬·동일 프롬프트(이미지 참조만 차이)** → RGB 유무만 격리.

---

## RQ2. VLM이 GCRL의 long-horizon 한계를 보완하는가?
**가설**: GCRL 단독은 **먼 목표(원거리·wide-swing) 도달이 어렵고**, VLM의 sub-goal 분해가 이를 해결.

| 항목 | 설계 |
|---|---|
| **과업** | 단일 goal reaching + 장애물 (goal은 좌표로 제공) |
| **변인(IV)** | **A(GCRL 단독, goal 직접)** vs **B(VLM+GCRL, sub-goal relay)** |
| **고정** | GCRL, VLM 모델·프롬프트, 시나리오, seed (paired) |
| **주지표** | **Success** (+ safe-success, collision) |
| **분석** | goal을 **거리·방위별 구간**(near/mid/far, front/side/wide)으로 나눠 **원거리에서 B−A Δ가 큰지** |
| **예상** | 원거리/wide에서 A 실패 → B가 회복 (long-horizon 분해 효과) |
| **필요/상태** | `Evaluate/vlm_guided/mission_eval.py` **보유**(A/B paired) → **fixed GCRL + 신 분포 + feasibility seed로 재실행**. n≥50 |

**엄격성 포인트**: A와 B는 **완전히 동일한 장애물·goal**(paired) → 유일 차이가 sub-goal 유무. 프롬프트 고정.

---

## RQ3. 새로운 환경에서도 작동하는가? (ID vs OOD 일반화)
**가설**: 학습 때 못 본 장애물 구성·배치·객체에도 성능 유지(일반화).

| 항목 | 설계 |
|---|---|
| **변인(IV)** | 분포: **ID**(학습 분포) vs **OOD** 3종 |
| **OOD (a)** | 장애물 **개수 증가** (n_obstacles 3 → 5~7) |
| **OOD (b)** | **새 배치 패턴** (예: goal 주변 밀집 / corridor 폭·t 변경 / 후면 포함 등 학습 때와 다른 규칙) |
| **OOD (c)** | **새 객체 종류·크기** (더 큰 장애물·다른 종횡비, 또는 non-box 형상) |
| **고정** | GCRL, VLM, 프롬프트 |
| **지표** | **성능 유지율 = OOD성능 / ID성능** (success·safe-success 각각) |
| **예상** | 유지율 높음(예 ≥0.8) → 강건. 하락 구간 분석으로 한계 규명 |
| **필요/상태** | env에 **OOD 변형 파라미터**(n·배치·크기) 추가 → ID/OOD 각각 A/B·VLM 평가. seed 분리 |

**엄격성 포인트**: ID/OOD **동일 GCRL·VLM·프롬프트**, 분포만 변경. 유지율은 **paired 불가**(분포 다름)라 각 분포 n≥50 독립 측정 + 신뢰구간.

---

## RQ4. VLM 모델 규모가 전체 성능을 좌우하는가?
**가설**: 상위 planner(더 큰 VLM)의 인지·계획 능력이 전체 프레임워크 성능을 결정.

| 항목 | 설계 |
|---|---|
| **변인(IV)** | VLM 모델만 **haiku → sonnet → opus** |
| **고정** | GCRL, 프롬프트, 시나리오, seed (paired) |
| **지표** | ① **인지 정확도**(recall·precision·위치/방위/높이 오차·환각) ② **회피 성능**(safe-success·collision) |
| **예상** | 모델 규모에 따라 **단조 증가**(인지·성능 both) |
| **필요/상태** | 인지분석 스크립트·A/B 러너 **보유** → **fixed GCRL·feasibility seed로 재실행**. 이미 확보된 경향: 인지 opus>sonnet>haiku(위치 0.13 vs 0.47m), clean셋 safe 0.41>0.29>0.12 |

**엄격성 포인트**: 3모델 **동일 seed·프롬프트** → VLM만 변인. 인지↔성능 **decoupling**(오염셋에선 역전)을 clean셋으로 해소했음을 명시.

---

## RQ5. 프롬프트 설계가 회피 성능에 얼마나 기여하는가?
**가설**: 안전한 sub-goal 생성을 유도하는 프롬프트가 GCRL 회피를 크게 개선.

| 항목 | 설계 |
|---|---|
| **변인(IV)** | 프롬프트만 **Baseline** vs **Smooth** |
| Baseline | 목표 도달 + 회피만 지시 |
| Smooth | ① 장애물 영역 명시 식별 ② 단계적 경로분해(촘촘 sub-goal) ③ 경로 충돌검사·안전여유 ④ **여유 위반 sub-goal 보정**(§6 여유확대 포함) |
| **고정** | VLM 모델, GCRL, 시나리오, seed (paired) |
| **지표** | safe-success·collision (+ sub-goal 수 = 출력구조 변화) |
| **예상** | Smooth가 도달·안전 개선. 단 **약한 모델일수록 이득 큼**(diminishing returns) |
| **필요/상태** | Baseline·Smooth 프롬프트 **보유** → **§6 여유확대 최신본 포함 재실행**(이전 smooth 결과는 §6 전). 이미 확보 경향: haiku safe 0.12→0.41(+0.29), sonnet/opus +0.06 |

**엄격성 포인트**: 프롬프트 **파일만 교체**(VLM·GCRL·seed 동일) → 프롬프트 순수 효과. haiku+smooth ≈ opus+baseline(능력 대체) 강조.

---

## 실행 스케줄 (권장 순서)

| 단계 | RQ | 작업 | 재사용 | 신규 |
|---|---|---|---|---|
| **S0** | — | 유효 seed set 확정(신 분포·feasibility, ID/OOD 분리), fixed GCRL 연결 | screen_valid | seed set 고정 |
| **S1** | RQ2,4,5 | 단일 러너로 **A/B × {haiku,sonnet,opus} × {baseline,smooth}** 대량 실행(캐시) | mission_eval | fixed GCRL·신분포 재실행 |
| **S2** | RQ5 | §6 여유확대 smooth 포함 → collision 개선 확인 | smooth prompt | §6 반영 |
| **S3** | RQ3 | OOD 변형(n·배치·크기) env → ID/OOD 평가 | mission_eval | OOD 파라미터 |
| **S4** | RQ1 | 상차 씬 Full vs No-RGB → 목표 식별 | mission_scene | No-RGB 프롬프트·러너 |
| **S5** | — | 정량표·GIF·인지 산점도 정리 → 논문 figure/table | — | 시각화 |

**효율화**: RQ2·4·5는 **동일 A/B 프레임워크**라 seed×model×prompt를 **한 번의 그리드 실행**으로 모두 커버(계획 캐시 공유). RQ1·RQ3만 별도 setup.

---

## 논문 표/그림 매핑 (예상 산출물)
- **Table 1**: RQ2 — 구간별(near/mid/far) A vs B success (long-horizon 효과)
- **Table 2**: RQ4 — 모델별 인지 정확도 + safe-success (단조 증가)
- **Table 3**: RQ5 — baseline vs smooth × 3모델 (프롬프트 기여)
- **Table 4**: RQ3 — ID vs OOD(a/b/c) 유지율
- **Table 5**: RQ1 — Full vs No-RGB 목표 식별·success
- **Fig**: rollout GIF 정성(수직통과/우회), 인지 오차 산점도, Pareto(있으면)

## 통계·엄격성 체크리스트
- [ ] 모든 회피 RQ: **feasibility 스크리닝**된 유효 시나리오만
- [ ] paired RQ(2,4,5): **동일 seed** 확인, win/loss 카운트 병기
- [ ] n≥50, 가능하면 **신뢰구간/부트스트랩**
- [ ] VLM 계획 **캐시**(재현성) + rate-limit 대비 workers 2~4
- [ ] fixed GCRL(`GCRL_fixed.zip`) 전 실험 동일 사용
- [ ] OOD: 학습 분포와 **명확히 구분되는** 변형만 OOD로 규정

---

## 부록: 지금까지 확립된 예비 결과 (재실행 시 참조)
- RQ2/4/5의 A/B·모델·프롬프트 경향은 clean 17-seed에서 이미 관찰(RESULTS.md). 단 **fixed GCRL·신 분포·§6 프롬프트·확대 seed로 재실행**해 논문 수치를 확정.
- 방법론적 핵심: **평가셋 오염(91%)** 발견 → feasibility 스크리닝이 결론을 좌우 → 논문의 방법론 기여로 서술 가능.
