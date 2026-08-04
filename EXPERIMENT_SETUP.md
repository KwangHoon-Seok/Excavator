# Experiment Environment Setup

VLM(high-level planner) + GCRL(low-level controller) 계층형 굴착기 임무계획 실험을
**새 머신(예: Elice Cloud)** 에서 재현하기 위한 환경 구축 가이드.

---

## 0. TL;DR (빠른 셋업)

```bash
# 1) 코드 받기
git clone https://<TOKEN>@github.com/KwangHoon-Seok/Excavator.git qwen-vlm
cd qwen-vlm

# 2) 절대경로 치환 (하드코딩된 /home/aimlab/qwen-vlm → 현재 위치)
NEWROOT=$(pwd)
grep -rl '/home/aimlab/qwen-vlm' --include=*.py --include=*.xml . \
  | xargs sed -i "s#/home/aimlab/qwen-vlm#${NEWROOT}#g"

# 3) conda 환경
conda create -n seok python=3.10 -y && conda activate seok
pip install -r requirements.txt          # (§5에서 생성)

# 4) 헤드리스 렌더링 백엔드
export MUJOCO_GL=egl

# 5) 스모크 테스트 (§8)
python -c "import mujoco, stable_baselines3, torch; print('ok', torch.cuda.is_available())"

# 6) 실험 실행 (§9)
python Evaluate/vlm_guided/mission_eval.py --vlm_model haiku --n_seeds 20
```

---

## 1. 시스템 구성 — 무엇이 어디서 도는가

| 모듈 | 역할 | 연산 요구 | 비고 |
|---|---|---|---|
| **VLM planner** | RGB + gridmap → 서브골 시퀀스 JSON | (백엔드에 따라 다름) | §6 참조 |
| **Interface** | JSON 파싱, 서브골 스위칭, 링크 clearance | CPU | — |
| **GCRL controller** | 서브골 도달 저수준 제어 (SAC+HER) | **CPU 충분** (정책 3.2MB) | `v4base.zip` |

> **핵심:** GCRL 컨트롤러는 작아서 CPU로도 충분히 평가된다. **GPU가 필요한 건 오직 VLM**
> (그것도 로컬 Qwen을 쓸 때만). Claude CLI 백엔드를 쓰면 GPU가 아예 필요 없다.

---

## 2. 하드웨어 타깃

| 시나리오 | 권장 |
|---|---|
| VLM = **Claude CLI** | GPU 불필요. CPU 인스턴스면 충분 |
| VLM = **Qwen2.5-VL-3B** (로컬 경량) | GPU 8–16GB (RTX 2080/3090/A10 등) |
| VLM = **Qwen 30B급 MoE** (논문 목표) | **A100 80GB ×1** 권장 (bf16 추론) |

현재 개발 PC: RTX 2080 SUPER **8GB** — 30B급은 **불가**, 3B 4bit는 가능.

---

## 3. 코드/자산 받기

원격: `https://github.com/KwangHoon-Seok/Excavator.git` (사용자 `KwangHoon-Seok`)

### push해야 반영되는 것 (개발 PC에서 먼저)
- ✅ `excavatorURDF/` — 메시/URDF 70개 파일 (이미 커밋 완료: `0010d7e`)
- ⚠️ **`Model/checkpoints/Version4/v4base.zip`** — `.gitignore`가 `Model/checkpoints/`를
  제외하므로 **강제 추가** 필요:
  ```bash
  git add -f Model/checkpoints/Version4/v4base.zip
  git commit -m "Add v4base checkpoint for eval"
  git push origin master
  ```

### 받기 (새 머신에서)
```bash
git clone https://<TOKEN>@github.com/KwangHoon-Seok/Excavator.git qwen-vlm
```
로그인/PAT 방법은 별도 안내(§Appendix) 참조.

---

## 4. 절대경로 치환 (필수)

다음 위치에 `/home/aimlab/qwen-vlm`가 하드코딩되어 있음:
- `excavator.xml` — 메시 `.stl` 경로 **14곳**
- `Model/Version4/excavator_env.py:17` — `ROOT = "..."`
- `VLM/mission_scene.py:17` — `ROOT = "..."`

한 번에 치환:
```bash
NEWROOT=$(pwd)                 # 예: /root/qwen-vlm
grep -rl '/home/aimlab/qwen-vlm' --include=*.py --include=*.xml . \
  | xargs sed -i "s#/home/aimlab/qwen-vlm#${NEWROOT}#g"
# 확인 (0이 나와야 함)
grep -rl '/home/aimlab/qwen-vlm' --include=*.py --include=*.xml . | wc -l
```

> 클론 위치를 `/home/aimlab/qwen-vlm`로 똑같이 맞추면 이 단계는 생략 가능.

---

## 5. Conda 환경 (`seok`)

개발 PC에서 검증된 버전 (Python 3.10.20):

`requirements.txt`:
```
mujoco==3.10.0
stable_baselines3==2.9.0
gymnasium==1.3.0
torch==2.6.0            # cu124 빌드 (아래 참고)
numpy==2.2.6
matplotlib==3.10.9
opencv-python==4.13.0.92
pillow==11.3.0
# --- VLM = 로컬 Qwen을 쓸 때만 ---
transformers==5.12.1
accelerate==1.14.0
qwen-vl-utils
bitsandbytes           # infer.py의 4bit 로딩용 (선택)
```

설치:
```bash
conda create -n seok python=3.10 -y && conda activate seok
# torch는 CUDA 12.4 빌드로:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **CUDA:** 개발 환경은 torch 2.6.0 **+cu124**. 클라우드 드라이버가 CUDA ≥12.4를
> 지원하는지 `nvidia-smi`로 확인. 하위면 cu121 빌드로 낮추면 됨.

---

## 6. VLM 백엔드 선택 (3가지)

### 옵션 A — Claude CLI (기본값, GPU 불필요) ← 현재 파이프라인 default
- 스크립트: `VLM/VLM_infer.py` → `claude -p --model haiku|sonnet|opus` 호출
- `mission_eval.py --vlm_model haiku`(기본)가 이걸 사용
- **요구:** `claude` CLI 설치 + 로그인 (별도 API 키 불필요, Claude Code 인증 재사용)
- 장점: 셋업 최소, GPU 0. 단점: 네트워크/CLI 로그인 의존

### 옵션 B — Qwen2.5-VL-3B 로컬 (경량 GPU)
- 스크립트: `VLM/infer.py` (`MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"`)
- 요구: `transformers`, `accelerate`, `qwen-vl-utils`, (4bit면 `bitsandbytes`)
- GPU 8GB에서 4bit로 구동 가능

### 옵션 C — Qwen 30B급 MoE (논문 목표, A100 80GB)
- `infer.py`의 `MODEL_ID`를 목표 모델(예: `Qwen/Qwen3-VL-30B-A3B-Instruct`)로 교체
- **vLLM 권장** (처리량/메모리 효율):
  ```bash
  pip install vllm
  # 서버: vllm serve <MODEL_ID> --limit-mm-per-prompt image=2
  ```
- 또는 transformers bf16 직접 로딩 (A100 80GB 가정)

> **RQ 실험 정합성:** 백엔드를 바꿔도 **입력(RGB+gridmap)·프롬프트·출력 JSON 스키마는 동일**
> 하게 유지해야 A/B 및 RQ 결과가 비교 가능하다.

---

## 7. 체크포인트 (GCRL 컨트롤러)

- **사용 파일:** `Model/checkpoints/Version4/v4base.zip`
- **관측 공간 (Dict, HER):**
  - `achieved_goal (3,)`, `desired_goal (3,)`, `observation (20,)`
  - `observation 20 = qpos4(정규화) + qvel4(정규화) + clearance12`
  - **action (4,)** = swing/boom/stick/bucket 증분
- ⚠️ **`obstacle_training=True`(20-dim)로 학습됨.** 평가 시 env도 반드시 20-dim으로
  띄워야 함. `obstacle_training=False`(8-dim)로 로드하면 shape mismatch로 실패.

---

## 8. 스모크 테스트 (실험 전 최소 확인)

```bash
conda activate seok
export MUJOCO_GL=egl        # 헤드리스 렌더 필수

# (1) 핵심 임포트 + CUDA
python -c "import mujoco,stable_baselines3,torch,gymnasium; \
print('mujoco',mujoco.__version__,'sb3',stable_baselines3.__version__,'cuda',torch.cuda.is_available())"

# (2) XML 로드 (경로 치환 검증)
python -c "import mujoco; mujoco.MjModel.from_xml_path('excavator.xml'); print('xml ok')"

# (3) 체크포인트 관측 차원 확인 (20 나와야 함)
python - <<'PY'
import zipfile,json,base64,pickle
d=json.loads(zipfile.ZipFile('Model/checkpoints/Version4/v4base.zip').read('data'))
s=pickle.loads(base64.b64decode(d['observation_space'][':serialized:']))
print({k:v.shape for k,v in s.spaces.items()})
PY

# (4) 헤드리스 렌더 1프레임 (gridmap/RGB 생성 경로 검증)
python -c "import mujoco,numpy as np; m=mujoco.MjModel.from_xml_path('excavator.xml'); \
d=mujoco.MjData(m); r=mujoco.Renderer(m,480,640); r.update_scene(d); \
print('render', r.render().shape); r.close()"
```

---

## 9. 실험 실행 & RQ 매핑

### 엔트리포인트
`Evaluate/vlm_guided/mission_eval.py` (주요 인자):

| 인자 | 기본 | 의미 |
|---|---|---|
| `--vlm_model` | `haiku` | Claude 모델 (haiku/sonnet/opus) 또는 백엔드 |
| `--n_seeds` | 20 | 평가 seed 수 |
| `--seed0` | 300000 | 시작 seed |
| `--workers` | 4 | VLM 병렬 호출 수 |
| `--timeout` | 1200 | VLM 호출당 타임아웃(초) |
| `--sub_steps` | 150 | B: 서브골당 최대 스텝 |
| `--ckpt` | `Version4/v4base.zip` | GCRL 체크포인트 |
| `--refresh` | off | VLM 계획 캐시 무시하고 재호출 |
| `--no_gif` | off | 롤아웃 GIF 저장 끄기 |

예시:
```bash
export MUJOCO_GL=egl
python Evaluate/vlm_guided/mission_eval.py --vlm_model haiku --n_seeds 20 --workers 4
```

### RQ 매핑
per-RQ 상세 실험 계획(가설·조건·씬·실행·판정)은 **§11** 참조. metric 정의는 `EXPERIMENTS.md §2`.
- **RQ1** 의미 인식 필요성 — Full vs No-RGB (gridmap ambiguous)
- **RQ2** 장기 임무 — A(단일 goal) vs B(VLM 서브골 릴레이) + 충돌
- **RQ3** GCRL OOD에서 VLM 보조 효과
- **RQ4** VLM 모델 스케일 (haiku/sonnet/opus, Qwen 크기)
- **RQ5** 프롬프트 설계

---

## 10. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `GLFW`/`EGL` 렌더 에러 (헤드리스) | `export MUJOCO_GL=egl` (또는 `osmesa`) |
| framebuffer width 초과 | 렌더 해상도 ≤ 1280 |
| 메시 `.stl` not found | §4 경로 치환 누락 — `grep -rl /home/aimlab` 재확인 |
| 체크포인트 로드 shape mismatch | env를 `obstacle_training=True`(20-dim)로 띄웠는지 확인 |
| `claude: command not found` | 옵션 A 사용 시 CLI 미설치/미로그인 — §6 |
| CUDA 버전 불일치 | torch 빌드를 드라이버 CUDA에 맞춰 재설치 (cu124/cu121) |

---

## 11. RQ별 상세 실험 계획

> **공통 통제 (모든 RQ 고정):**
> - GCRL 컨트롤러 = `Version4/v4base.zip` **동결**(재학습 없음), env `obstacle_training=True`(20-dim).
> - 씬당 seed 고정 생성 → 모든 조건(arm)이 **동일 씬**을 공유(paired 비교).
> - metric 정의는 `EXPERIMENTS.md §2` (MSR/CR/MC/SRR/PL/TTC/PV/SC/IA).
> - 충돌/clearance = 매 스텝 `mujoco.mj_geomDistance(arm·bucket ↔ obstacle·truck)`, `min<0`=충돌, `min<0.2m`=unsafe.
> - 통계: rate는 Wilson CI, 조건 간 MSR은 동일 씬 paired → **McNemar**. VLM은 greedy 디코딩(재현성).
> - 기본 규모: **씬 N=100/조건**, VLM 백엔드 결정(§6)은 RQ4를 제외하고 **1종으로 고정**.

---

### RQ1 — 의미 인식 필요성 (Semantic Ablation: RGB 제거)

**질문.** VLM의 의미 인식(색/재질로 굴착 대상 vs 장애물 구분)이 **필수**인가, 아니면
height-map(gridmap)만으로 충분한가?

| 항목 | 내용 |
|---|---|
| **가설 H1** | gridmap이 **모호(ambiguous)** 한 씬에서 **Full(RGB+gridmap) ≫ No-RGB(gridmap only)**. No-RGB는 target/obstacle을 구분 못 해 오굴착·충돌 급증. |
| **독립변수** | VLM 입력 모달리티: `{Full = RGB+gridmap}` vs `{No-RGB = gridmap only}` |
| **통제** | 동일 프롬프트·동일 VLM·동일 GCRL·동일 씬. 오직 RGB 이미지 제공 여부만 토글 |
| **종속변수** | **SC**(의미 정확도, 주지표), **IA**(지시 달성), MSR, CR |
| **핵심 씬 조건** | **matched-geometry**: 동일한 흙덩이 클러스터를 2개 배치 → 하나는 **굴착 대상(갈색)**, 하나는 **장애물(회색)**. 기하가 같아 **gridmap상 두 blob이 동일(모호)**, RGB만 색으로 구분 가능. (이미 검증: `figures/rq1/rgb2.png` vs `gridmap2.png`) |
| **절차** | ① target 좌표를 주지 않고 VLM이 RGB로 대상을 식별하게 함 → ② dig subgoal 생성 → ③ GCRL 실행. Full/No-RGB 각각 N=100 씬. |
| **판정** | H1 성립 = No-RGB의 **SC가 우연 수준(≈50%, 두 blob 중 랜덤)** 으로 붕괴 & IA·MSR 유의하게 낮음(McNemar p<0.05). Full은 SC≈1.0. |
| **선행 확인(필수)** | 본 실험 전 소규모(N=10)로 **No-RGB가 실제로 대상 식별에 실패**하는지 확인. 실패하지 않으면 gridmap이 이미 충분 → RQ1 재설계. |
| **산출물** | 표(Full vs No-RGB: SC/IA/MSR/CR) + 정성 그림(같은 gridmap, 다른 RGB → 다른 판단) |

```bash
# 예시 (플래그명은 mission_eval에 --no_rgb 추가 필요 — §구현)
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --n_seeds 100          # Full
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --n_seeds 100 --no_rgb # No-RGB
```

---

### RQ2 — 장기 임무 분해 (단일 goal A vs VLM 서브골 릴레이 B)

**질문.** 장애물이 있는 장기 임무에서 **VLM 서브골 시퀀스(B)** 가 **단일 최종 goal(A)** 대비
도달·안전을 개선하는가?

| 항목 | 내용 |
|---|---|
| **가설 H2** | 장애물 우회가 필요한 씬에서 **B(서브골 릴레이) ≫ A(단일 goal)**: MSR↑, CR↓. A는 직선 접근으로 장애물 관통. |
| **독립변수** | 계획 구조: `{A = 최종 goal 1점}` vs `{B = VLM up-over-down 서브골 시퀀스}` |
| **통제** | 동일 GCRL·동일 총 스텝예산(`--max_steps_B`=A와 동일)·동일 씬 |
| **종속변수** | **MSR**(주지표), **CR**, **MC**(5th-pct), SRR, PL, TTC |
| **씬 조건** | 장애물 1–2개가 pile→truck 직선 경로를 가로막는 canonical + 지터(장애물 xy±0.2, h±0.1) |
| **절차** | mission_eval의 A/B 두 모드 각각 N=100 동일 씬. B는 서브골당 `--sub_steps=150` 가드. |
| **판정** | H2 성립 = paired MSR 차이 McNemar p<0.05, CR(B)≪CR(A). PL은 B가 다소 ↑(우회 비용) 허용. |
| **주의** | B의 PL이 과도하면 "과우회" — PL/TTC로 감지해 프롬프트(RQ5)로 보정. |
| **산출물** | A/B metric 표 + 대표 롤아웃 GIF(A=관통, B=회피) |

```bash
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --n_seeds 100            # B(서브골)
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --n_seeds 100 --single_goal  # A(구현 필요)
```

---

### RQ3 — GCRL OOD에서 VLM 보조 효과

**질문.** GCRL 컨트롤러의 **학습 분포 밖(OOD)** 상황에서도, VLM 서브골이 성능을
**개선/유지**시키는가? (※ VLM의 OOD 우월성 주장이 아니라, **동결된 GCRL을 VLM이 얼마나 보조**하는지)

| 항목 | 내용 |
|---|---|
| **가설 H3** | GCRL 단독(A)이 저하되는 OOD 씬에서도 **B(VLM 서브골)가 A 대비 성능 격차를 유지/확대**. VLM이 GCRL을 도달 가능·안전한 국소 목표로 유도. |
| **독립변수** | (계획 A vs B) × (씬 분포 in-dist vs OOD) |
| **OOD 축** | GCRL 학습과 다른: 장애물 **개수 0/1/2/3**, 방위 ±90° 전개, 높이 0.5–3.5m, 트럭 좌/우, 이중 코리도/초고 장애물 |
| **종속변수** | MSR, CR, SRR (in-dist 대비 **degradation Δ**) |
| **통제** | GCRL 동결(추가 학습 없음이 곧 OOD 조건). VLM은 zero-shot(고정) |
| **판정** | H3 성립 = OOD에서 `MSR(B)−MSR(A)` 격차가 in-dist 이상으로 유지(비열등). |
| **프레이밍 주의** | "VLM이 FSM보다 OOD 강함" **주장 금지**(VLM은 장애물 미학습). "per-scene 엔지니어링 없이 동결 컨트롤러를 보조"로 서술. |
| **산출물** | in-dist vs OOD × A/B 4셀 표 + OOD 축별 degradation 곡선 |

```bash
# 씬 분포를 seeds_file/scene_gen으로 OOD 구성 (§구현: scene_gen.py)
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --seeds_file ood_seeds.txt
```

---

### RQ4 — VLM 모델 스케일

**질문.** VLM 용량(모델 크기/등급)이 계획 품질에 미치는 영향과, **경량 모델의 실용 하한**은?

| 항목 | 내용 |
|---|---|
| **가설 H4** | 큰 VLM일수록 SC·MSR↑이나, **일정 등급 이상에서 수렴(diminishing returns)** → 경량 모델도 실용 가능. |
| **독립변수** | VLM 백엔드/크기: Claude `{haiku, sonnet, opus}` 및/또는 Qwen `{2.5-VL-3B, 30B-A3B}` |
| **통제** | 동일 프롬프트·씬·GCRL. `--vlm_model`만 변경 |
| **종속변수** | SC, MSR, CR + **비용축**(지연시간/토큰/실패율) |
| **씬 조건** | RQ2 canonical 세트(N=100) 재사용 → 동일 씬 paired |
| **판정** | 등급별 SC/MSR 단조성 + 수렴 지점 식별. 경량 모델의 절대 성능이 임계(예: MSR>0.7) 이상이면 "실용 하한" 근거. |
| **산출물** | 모델 스케일 vs (SC/MSR/지연) 곡선, Pareto(성능–비용) |

```bash
for M in haiku sonnet opus; do
  python Evaluate/vlm_guided/mission_eval.py --vlm_model $M --n_seeds 100 --out out_$M
done
```

---

### RQ5 — 프롬프트 설계

**질문.** 프롬프트 구조(scene interpretation / kinematic feasibility / path-segment safety /
structured output)의 개선이 계획 품질을 향상시키는가?

| 항목 | 내용 |
|---|---|
| **가설 H5** | 안전·kinematic 제약을 명시한 **구조화 프롬프트 > naive 프롬프트**: CR↓, PV↑, 과우회(PL) 완화. |
| **독립변수** | 프롬프트 변형: `{naive}` vs `{structured(현행)}` vs `{smooth(우회 완만화)}` |
| **통제** | 동일 VLM·씬·GCRL. `--prompt`로 텍스트만 교체 |
| **종속변수** | CR, MC, PV, PL(과우회), MSR, SC |
| **씬 조건** | RQ2 canonical + 안전 민감 씬(초고/근접 장애물) |
| **판정** | structured가 naive 대비 CR·PV 유의 개선. smooth는 PL↓ 확인하되 **diminishing returns 가능**(기존 관찰). |
| **참고** | `VLM/VLM_guidedGCRL_prompt.txt`, `VLM/VLM_guidedGCRL_smooth_prompt.txt` |
| **산출물** | 프롬프트 변형 × metric 표 + 대표 궤적(과우회 완화 전/후) |

```bash
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --prompt VLM/VLM_guidedGCRL_prompt.txt
python Evaluate/vlm_guided/mission_eval.py --vlm_model <fixed> --prompt VLM/VLM_guidedGCRL_smooth_prompt.txt
```

---

### 실행 순서 (권장)
1. **선행 확인** — RQ1 No-RGB 소규모(N=10) 실패 검증 → RQ1 유효성 gate.
2. **RQ2** 파이프라인 관통(A/B) — mission_eval 메인 러너 검증.
3. **RQ1** 본실험 → **RQ5**(프롬프트로 CR/PL 개선) → **RQ4**(스케일) → **RQ3**(OOD).
4. 각 RQ 결과를 `EXPERIMENTS.md` 기대결과표에 반영.

### 구현 필요 (mission_eval 확장)
| 플래그/모듈 | 용도 | RQ |
|---|---|---|
| `--no_rgb` | RGB 입력 토글 | RQ1 |
| `--single_goal` (A 모드) | 서브골 없이 최종 goal 1점 | RQ2 |
| `Evaluate/scene_gen.py` | canonical/OOD 씬 seed 생성 | RQ2/3 |
| `Evaluate/collision.py` | 궤적 clearance/충돌 검출 | 전 RQ (CR/MC) |
| metric CSV 통합 | `EXPERIMENTS.md §8` 스키마 | 전 RQ |

---

## Appendix — GitHub 터미널 로그인 (요약)

```bash
# GitHub CLI (권장)
gh auth login          # GitHub.com → HTTPS → 브라우저(device code) 또는 토큰

# 또는 PAT + credential helper
git config --global credential.helper store
git clone https://github.com/KwangHoon-Seok/Excavator.git qwen-vlm
#   Username: KwangHoon-Seok / Password: <PAT>
```
PAT는 fine-grained + Excavator repo + Contents Read(클론)/Read·Write(push)로 최소권한.
