# Qwen-VLM-guided GCRL 실험 (`Experiment/Code`)

로컬 **Qwen2.5-VL**(HuggingFace)을 상위 플래너로 써서, RGB cabin 이미지 + 극좌표 color-height
grid map + system prompt + 언어 지시를 입력받아 **서브골 시퀀스(JSON)**를 만들고, 그걸
`interface.parse_plan`으로 변환해 학습된 **GCRL(SAC+HER) 제어기**가 순차 추종하게 하는 A/B 실험.

기존 Claude-CLI 파이프라인(`VLM/VLM_infer.py` + `Evaluate/vlm_guided/mission_eval.py`)의 VLM 단만
로컬 Qwen으로 교체한 것이다. 씬 렌더·롤아웃·metric·GIF는 `Evaluate/vlm_guided/`를 그대로 재사용한다.

```
env.reset(seed) ─▶ render_scene.render_current ─▶ cabin_init.png + gridmap.png + gridmap_legend.txt + scene.json
                                                          │
                                                          ▼
        qwen_planner.infer (Qwen2.5-VL: system prompt + 2 images + instruction + goal)
                                                          │  plan.json / goals.json
                                                          ▼
                        interface.parse_plan ─▶ goals = [[x,y,z], ...]
                                                          │
                                                          ▼
        rollout_B: env._desired = wp; env._get_obs(); SAC.predict … (서브골 relay, 마지막=실제 goal)
```

## 파일
- **`qwen_planner.py`** — 로컬 Qwen2.5-VL 추론 모듈. 모델 싱글턴 로드 + `infer(...)`(=`VLM_infer.infer` 드롭인).
  단독 실행으로 단일 씬 계획 테스트 가능.
- **`run_experiment.py`** — A/B paired 실험 러너(렌더 → Qwen 계획 → GCRL 롤아웃 → CSV/GIF).
- **`requirements.txt`** — 의존성(설치법 주석 포함).

## 설치 (1회)
시스템 드라이버는 CUDA 12.2(A100 80GB×2)라 **cu124 torch**가 필요하다:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r Experiment/Code/requirements.txt
# 확인: True 2 가 떠야 함
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 사용법
저장소 루트(`/home/elicer/Excavator`)에서 실행한다.

> 디스플레이 없는 서버라면 MuJoCo 오프스크린 렌더에 소프트웨어 GL이 필요하다.
> `run_experiment.py`는 자동으로 `MUJOCO_GL=osmesa`를 설정하지만, `render_scene.py`를 **단독**으로
> 부를 때는 환경변수를 직접 줘야 한다: `MUJOCO_GL=osmesa python ...`
> (1회 설치: `sudo apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3`)

**1) 단일 씬 계획 테스트** (Qwen 로드 + JSON 서브골 확인):
```bash
# 먼저 씬 하나 렌더(RGB+gridmap+scene.json 생성)
MUJOCO_GL=osmesa python Evaluate/vlm_guided/render_scene.py --seed 300000 --out Experiment/Code/out/seed300000
# Qwen으로 계획(goal은 scene.json에서 자동 로드)
python Experiment/Code/qwen_planner.py \
    --rgb  Experiment/Code/out/seed300000/cabin_init.png \
    --grid Experiment/Code/out/seed300000/gridmap.png \
    --out  Experiment/Code/out/seed300000
# → out/seed300000/plan.json, goals.json
```

**2) A/B 스모크(2 seed, GIF 끔)**:
```bash
python Experiment/Code/run_experiment.py --n_seeds 2 --no_gif
```

**3) 본 실험**(검증된 valid seed 세트):
```bash
python Experiment/Code/run_experiment.py --seeds_file Evaluate/vlm_guided/valid_seeds.json
```

**더 큰 Qwen으로 교체**(코드 수정 불필요, A100 2장에 자동 분산):
```bash
python Experiment/Code/run_experiment.py --qwen_model Qwen/Qwen2.5-VL-32B-Instruct --n_seeds 5 --no_gif
```

## 출력
`Experiment/Code/out/<model-tag>/`:
- `seed<S>/` — `cabin_init.png`, `gridmap.png`, `gridmap_legend.txt`, `scene.json`, `plan.json`,
  `goals.json`, (`rollout_A.gif`, `rollout_B.gif`)
- `results_<model-tag>.csv` — seed별 paired metric(A/B success·collision·safe_success·dist, 구역)
- 콘솔: A vs B vs Δ 요약표 + 방위/거리 구역별 safe_success

## metric
- **A (baseline)**: GCRL이 실제 goal로 직접 이동.
- **B (Qwen-guided)**: Qwen 서브골 시퀀스를 relay(각 서브골 `‖tip−wp‖ < eps` 도달 또는 `--sub_steps`
  소진 시 다음, 마지막에 실제 goal 추가).
- **safe_success** = 최종 goal 도달 AND 전 구간 충돌 0회(`info["min_gap"] ≤ 0` 없음).
- 장애물 씬에서 **B의 collision ≤ A**, wide/far 구역에서 **B의 safe_success ≥ A**면 Qwen 플래너가
  회피·도달을 개선한다는 증거.

## 참고
- 기본 모델 `Qwen/Qwen2.5-VL-7B-Instruct`(bf16 ~16GB, A100 1장). 첫 실행 시 HF 허브에서 가중치 다운로드.
- system prompt/좌표 규약은 `VLM/VLM_guidedGCRL_prompt.txt`(reach+회피) 재사용. 좌표계는 GCRL 월드와 동일
  (x=우, y=전방≥0, z=높이) — 변환 불필요.
- 체크포인트 `Model/checkpoints/Version4/v4base.zip`(obstacle-aware, obs 20차원).
