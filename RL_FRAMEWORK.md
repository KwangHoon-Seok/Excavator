# Excavator GCRL 학습 프레임워크

VLM이 생성한 굴착기 임무계획을 GCRL(Goal-Conditioned RL)로 실행하는 프레임워크의 **학습 파트** 문서. 설계·실행법·reward 실험 기록(living doc).

> **현행 task: bucket waypoint 이동 (reaching).** VLM이 3D waypoint(들)를 생성하면, GCRL 정책이 bucket_tip을 그 점으로 보낸다. 목표는 바위와 무관하게 자유공간에서 샘플(`_sample_reachable_waypoint`). 바위 조작(move/pick/place + sticky-grasp + relay)은 **레거시**로 env에 남아있고 `fixed_type`으로 선택 가능.

## 1. 아키텍처
```
VLM(Qwen2.5-VL, 비전)  ──plan(목표 3D waypoint 시퀀스)──▶  3D goal (x,y,z)
                                                                  │
현재 상태(관절·FK·특권정보) ────────────────────▶  SAC + HER ──▶  4D 관절 액션(Δposition)
```
- VLM=지각·계획, GCRL=저수준 실행. 카메라 extrinsic은 관절 엔코더+FK로 추적(이동 센서는 calibration 아닌 kinematics 문제).
- goal = `[x, y, z]` (3D 위치). achieved=현재 bucket_tip, desired=목표 waypoint.

## 2. 폴더 구조
```
qwen-vlm/
├── excavator.xml, terrain.py          # 공유 자산(MuJoCo 모델 / hfield 굴착)
├── RL_FRAMEWORK.md                    # (이 문서)
├── view_excavator.py                  # 인터랙티브 뷰어(슬라이더 수동조작, 목표=초록 구)
├── Model/                             # 학습
│   ├── excavator_env.py   # ExcavatorRHEREnv (waypoint reaching)
│   ├── reward.py          # reward.dense(거리 기준) 사용
│   ├── train.py           # SAC+HER 학습 (config.yaml 기반)
│   ├── callbacks.py       # 주기 eval + metrics.csv + 그래프
│   ├── config.yaml        # 하이퍼파라미터/seed
│   ├── checkpoints/       # <run>.zip
│   └── logs/              # TensorBoard + <run>/metrics.csv
├── VLM/                              # VLM 지각·계획 (입력 생성 + 추론)
│   ├── gridmap.py        # 극좌표(θ,r) color-height 3D grid map 생성기(지형+물체, 이산 고정색)
│   ├── make_gridmap.py   # VLM 입력 페어 생성(RGB+gridmap 480x1280) → cabin_init.png/gridmap.png/npz/json + gridmap_legend.txt
│   ├── render_view.py    # cabin_cam(상단-우측) RGB 렌더 → VLM/out/cabin_view.png
│   ├── infer.py          # Qwen2.5-VL + SAM2 추론(RGB → sub-goal point)
│   ├── system_prompt.txt # VLM 시스템 프롬프트
│   └── out/              # VLM 입출력 산출물(gridmap.*, result_*.png)
└── Evaluate/              # 평가
    ├── evaluate.py        # 랜덤 waypoint 성공률/평균거리
    ├── visualize.py       # 뷰어 롤아웃 / GIF (목표=초록 구)
    ├── infer_waypoint.py  # 지정 waypoint(--wp x y z) 추론 + 뷰어/GIF
    └── plots/             # 학습곡선 PNG, GIF
```
(루트 초기 스캐폴드 `train_gcrl.py`/`eval_gcrl.py`는 Model/Evaluate 버전으로 대체됨.)

## 3. 환경 — `Model/excavator_env.py`
- **관측 Dict**: `observation`=`[qpos4, qvel4]`(정규화, 8차원), `achieved_goal`=`[tip xyz]`(3), `desired_goal`=`[waypoint xyz]`(3).
  - tip은 achieved_goal에 있어 observation에선 생략. 바위/force/onehot은 waypoint task에 불필요 → 제거.
- **액션**: `Box(-1,1,(4,))` → env에서 증분 역정규화 `tgt = clip(qpos + a·Δmax, lo, hi)`. MuJoCo ctrlrange는 물리(rad) 유지(안전 클램프). 정규화는 gym 층에만.
- **목표 샘플(원통 균일)**: init에서 FK로 도달 가능한 `(r,z)` 셀 envelope를 한 번 만들고, `_sample_reachable_waypoint`가 그 셀을 **(r,z)-균일**로 고른 뒤 θ를 360° 균일 → 반경·높이 띠를 고르게 학습. (Cartesian 균일은 면적∝r로 먼 goal 과대표집 → 원통 채택)
- **커리큘럼(r 기준)**: `set_curriculum(frac)`로 허용 반경 `curr_r`을 조절(0=최근접, 1=전체). 학습 시 `CurriculumCallback`이 성공률이 `curr_up`(0.7) 넘으면 `curr_inc`(0.2)만큼 확대 → 가까운 waypoint부터 점차 멀리. config `curriculum: true`로 on/off.
- **목표 지정**: `reset(options={"goal":[x,y,z]})`로 waypoint 직접 지정(시각화/추론/VLM 브리지).
- **시작 자세**: 항상 home `[0,-0.2,-0.8,0]`. **frame_skip** 20 (500Hz→25Hz 제어), **max_steps** 300.

### 성공 predicate (rule-based)
| task | desired_goal xyz | achieved xyz | 성공 |
|---|---|---|---|
| **waypoint** | 랜덤 도달점(FK) 또는 지정 goal | bucket_tip | ‖tip − goal‖ < ε(0.3m) |

## 4. SAC + HER
- **단일 goal-conditioned SAC**(SB3 `MultiInputPolicy` + `HerReplayBuffer`, future, k=4).
- 단일 stage. 매 에피소드 랜덤 waypoint를 목표로 bucket_tip reaching. HER가 달성 미래상태를 가짜 목표로 relabel → 표본효율↑.

## 5. Reward — `Model/reward.py`
거리 기준으로 통일: `compute_reward = reward.dense(ag, dg, info, eps)`.
- `dense(ag,dg) = -(d>ε) - 0.1·d`  (d = ‖tip − waypoint‖)
- `(achieved_goal, desired_goal, info)`만으로 계산 → HER가 목표 relabel 후 재계산해도 일관(HER 호환의 필수 제약).
- (reward.py에 sparse/guiding도 남아있으나 현재 미사용.)

## 6. 학습 추이 그래프
- TensorBoard: `Model/logs/<run>/` (`tensorboard --logdir Model/logs`).
- `Model/callbacks.py`가 주기 eval(성공률·평균거리·ep_reward)을 `metrics.csv` + `Evaluate/plots/<run>_curve.png`(success/dist/reward vs step)로 저장.

## 7. 실행법 (conda env `seok`)
```bash
# 학습 (config.yaml 기반, --run/--steps만 오버라이드)
conda run -n seok python Model/train.py --run waypoint --steps 80000
# 정량 평가(랜덤 waypoint 성공률)
conda run -n seok python Evaluate/evaluate.py --ckpt Model/checkpoints/waypoint.zip --n 30
# 시각화 (뷰어 / GIF) — 랜덤 waypoint 롤아웃
conda run -n seok python Evaluate/visualize.py --ckpt Model/checkpoints/waypoint.zip
conda run -n seok RENDER=gif python Evaluate/visualize.py --ckpt Model/checkpoints/waypoint.zip
# 지정 waypoint로 추론 (--wp x y z, 미지정시 랜덤)
conda run -n seok python Evaluate/infer_waypoint.py --ckpt Model/checkpoints/waypoint.zip --wp 0.3 3.0 0.1 --render viewer
```

## 8. 컨벤션 / 환경
- conda `seok`, **torch 2.6.0+cu124(GPU)**, gymnasium 1.3, sb3 2.9, matplotlib/tensorboard.
- (sb3 2.9는 torch≥2.8 핀 경고가 뜨지만 2.6에서 정상 작동 — torch는 VLM torchvision·드라이버 호환 위해 2.6 유지)

## 9. 현황 / 다음 단계
- ✅ env를 **waypoint reaching 전용으로 슬림화** (obs 24→8, goal 6→3, 바위/force/grasp/relay 제거). env_checker PASS, 학습 스모크 OK.
- ✅ config/train/evaluate/visualize/infer/view_excavator/문서 일괄 반영.
- 🔄 waypoint 모델 학습(80k) → 랜덤·지정 waypoint 추종 검증.
- ⬜ 실패사례 렌더 디버깅, VLM→waypoint 브리지.
- 🗄️ 바위 조작(pick/place/sticky-grasp)은 git 이력에 보존 — MuJoCo granular 한계로 보류.
