# Section 4. Low-Level Controller (GCRL)

> 논문 초안(대략적 내용). 실제 구현 근거: [Model/Version4/excavator_env.py](Model/Version4/excavator_env.py), [reward.py](Model/Version4/reward.py), [callbacks.py](Model/Version4/callbacks.py), [config.yaml](Model/Version4/config.yaml).
> 구성: **4.1 Goal-conditioned MDP → 4.2 SAC+HER → 4.3 학습 테크닉**. (MDP를 먼저 정의해야 HER의 sparse-reward 해결을 설명할 수 있음.)

상위 VLM 플래너가 생성한 3D waypoint $\mathbf g\in\mathbb R^3$를 추종하는 저수준 제어기를, **goal-conditioned RL(GCRL)** 로 학습한다. 제어기는 waypoint 추종과 함께 지면·자기충돌 및 장애물에 대한 **국소 반응 회피**를 담당한다(전역 경로는 플래너가 담당 — Sec. 3의 역할 분담).

---

## 4.1 Goal-Conditioned MDP 설계

제어기 학습을 goal-conditioned MDP $(\mathcal S,\mathcal A,P,r,\mathcal G)$로 정식화한다.

### State (관측)
관측은 Dict 형태이며 모두 정규화된다(스케일 통일 → sim-to-real·gradient 안정).

| 필드 | 차원 | 정의 |
|---|---|---|
| `observation` | 20 | 관절위치 $\tilde{\mathbf q}$(4) + 관절속도 $\tilde{\dot{\mathbf q}}$(4) + 링크별 clearance(12) |
| `achieved_goal` | 3 | 버킷 끝 $\mathbf p_{\mathrm{tip}}/R_{\mathrm{reach}}$ |
| `desired_goal` | 3 | 목표 waypoint $\mathbf g/R_{\mathrm{reach}}$ |

- 관절위치: $\tilde{\mathbf q}=(\mathbf q-\mathbf q_{\mathrm{mid}})/\mathbf q_{\mathrm{half}}\in[-1,1]^4$ (관절 한계로 정규화).
- 관절속도: $\tilde{\dot{\mathbf q}}=\mathrm{clip}(\dot{\mathbf q}/\mathbf v_{\max},-1,1)$, $\mathbf v_{\max}=\boldsymbol\Delta_{\max}/\Delta t$.
- **링크별 clearance(12)**: boom·arm·bucket 각 링크에서 최근접 장애물까지 $[\hat{\mathbf n}_\ell,\bar g_\ell]$ (Sec. 3 Interface에서 계산). $\bar g_\ell=\mathrm{clip}(\mathrm{gap}_\ell/R_{\mathrm{reach}},-1,2)$, 멀면($\bar g_\ell\ge2$) $\hat{\mathbf n}_\ell=\mathbf0$로 마스킹.
- 버킷 끝은 `achieved_goal`에 있으므로 `observation`에서 생략(중복 제거).
- 관측이 **베이스 프레임 기구학량 + 상대 clearance**로만 구성 → 씬 외형·절대좌표에 불변.

### Action
$\mathbf a\in[-1,1]^4$ (관절별 증분). position servo 목표로 역정규화:
$$\mathbf q_{\mathrm{tgt}}=\mathrm{clip}(\mathbf q+\mathbf a\odot\boldsymbol\Delta_{\max},\ \mathbf q_{\mathrm{lo}},\ \mathbf q_{\mathrm{hi}})$$
$\boldsymbol\Delta_{\max}=[0.12,0.03,0.05,0.08]$ rad(스텝당 최대 증분), 제어주기 25 Hz(시뮬 500 Hz, frame-skip 20).

### Goal-conditioning & 성공 술어
목표공간 $\mathcal G=\mathbb R^3$(작업공간 위치). achieved $=\mathbf p_{\mathrm{tip}}$, desired $=\mathbf g$. 성공:
$$\text{success}\iff \lVert\mathbf p_{\mathrm{tip}}-\mathbf g\rVert<\epsilon\quad(\epsilon=0.2\,\mathrm m)$$
정책은 $\pi(\mathbf a\mid \mathbf s,\mathbf g)$로 목표에 조건화되어, 단일 정책이 임의 waypoint를 추종한다.

### Reward — 분해 설계
sparse만으로는 신호가 희박하고, 순수 dense는 reward hacking·gradient 불균형을 유발한다. 이를 **5항 분해**로 구성한다(모두 $[0,1]$ 정규화 항):
$$r=\underbrace{-\mathbb 1[d>\epsilon]}_{\text{sparse}}\ -\ w_s\underbrace{(\theta_{\mathrm{err}}/\pi)^2}_{\text{swing(방위)}}\ -\ w_r\underbrace{\mathrm{clip}\!\big(\tfrac{r_{\mathrm{err}}^2+z_{\mathrm{err}}^2}{\text{REACH\_NORM}},0,1\big)}_{\text{reach(평면)}}\ -\ w_{\mathrm{rate}}\underbrace{\tfrac{\lVert\mathbf a-\mathbf a_{\mathrm{prev}}\rVert^2}{4}}_{\text{action-rate}}\ -\ w_{\mathrm{ttc}}\underbrace{e^{-\mathrm{TTC}/\tau}}_{\text{장애물 TTC}}$$
- **sparse**: 도달 전 매 스텝 $-1$ → 최단 도달 유도, shaping 없어 hacking 불가.
- **swing/reach 분리**: 방위(선회)와 평면(반경·높이) 오차를 분리해 gradient 균형.
- **action-rate**: 크기가 아닌 **변화량**을 벌 → 빠른 이동 허용, 채터링만 억제.
- **TTC(장애물)**: 충돌 임박도 $e^{-\mathrm{TTC}/\tau}\in[0,1]$ (4.2에서 계산). 관통 시 1.
- 가중치(v4base): $w_s{=}w_r{=}0.5,\ w_{\mathrm{rate}}{=}0.1,\ w_{\mathrm{ttc}}{=}0.2,\ \tau{=}1.5$.

**HER-호환 제약(핵심 설계)**: 보상은 오직 $(\text{ag},\text{dg},\text{info})$로만 계산해야 한다. HER가 transition의 desired goal을 '달성 상태'로 relabel한 뒤 같은 함수로 재계산하기 때문이다. 따라서 절대 상태·시간을 쓰지 않고, action-rate·TTC 같은 transition 속성은 `info`에 담아 전달한다(relabel에도 유효).

> **[Figure 4.1]** MDP 개요: 관측(qpos/qvel+clearance) → 정책 → 증분 액션 → waypoint 추종. achieved=tip, desired=waypoint.

---

## 4.2 학습 알고리즘: SAC + HER

### SAC (Soft Actor-Critic)
연속 제어·표본효율을 위해 **최대 엔트로피 actor-critic**인 SAC를 채택한다. 목적함수는 보상과 정책 엔트로피의 가중합:
$$J(\pi)=\sum_t\mathbb E\big[r(\mathbf s_t,\mathbf a_t)+\alpha\,\mathcal H(\pi(\cdot\mid\mathbf s_t,\mathbf g))\big]$$
- 엔트로피 항이 탐색을 촉진하고 조기 수렴을 막는다. 온도 $\alpha$는 target entropy에 맞춰 자동 조정(4.3에서 확장).
- **Goal-conditioned SAC**: actor·critic 입력에 목표를 concat(SB3 `MultiInputPolicy`) → $\pi(\mathbf a\mid\mathbf s,\mathbf g),\ Q(\mathbf s,\mathbf a,\mathbf g)$. 단일 정책이 전 작업공간 waypoint를 커버.
- 하이퍼파라미터(v4base): $\gamma{=}0.98$, buffer $2{\times}10^5$, batch 256, lr $3{\times}10^{-4}$.

### HER (Hindsight Experience Replay) — sparse reward 극복
sparse+goal-conditioned에서는 초기에 목표 도달이 거의 없어 보상이 항상 $-1$ → 학습 신호 부재. HER는 **실패 궤적의 목표를 실제 도달한 상태로 사후 relabel**하여, 그 transition을 성공($r{=}0$)으로 되살린다.
- 전략: `future`(같은 에피소드의 미래 달성 상태를 가짜 목표로), $k{=}4$개 relabel.
- 효과: 모든 궤적이 "무언가에는 도달한" 성공 사례가 되어 밀집 학습 신호 생성 → sparse+HER가 정확히 이 문제를 푼다.
- relabel 후 보상 재계산이 성립하려면 4.1의 **HER-호환 reward 제약**이 필수.

> **[Figure 4.2]** HER relabeling 도식: (a) 실패 궤적(목표 미도달, 전부 $-1$) → (b) 도달 지점을 가짜 목표로 relabel → (c) 해당 transition이 성공($0$)으로 전환되어 신호 발생.

---

## 4.3 효율·일반화를 위한 학습 테크닉

### 4.3.1 Goal Normalization (design choice)
목표·tip을 작업반경 $R_{\mathrm{reach}}$로 정규화($\sim[-1,1]$)하면, 서로 다른 스케일(방위 vs 높이)의 오차가 균형을 이뤄 **gradient 불균형이 해소**된다. reach 항 정규화 상수 REACH_NORM도 측정된 envelope로 설정. (효과: goal-norm off 대비 학습 안정·성능 향상 — Sec. X ablation.)

### 4.3.2 Curriculum Learning (design choice)
쉬운 근거리 목표부터 점차 어려운 목표로 확대. 두 모드:
- **distance**: 시작 tip 기준 허용 반경 $r_{\mathrm{curr}}$을 넓힘(작은 reach → 큰 선회).
- **azimuth**: 방위폭 $\pm\theta$를 $\pm\theta_{\mathrm{start}}\to\pm\theta_{\mathrm{full}}$로 점진 확대(정면 도달 → 넓은 선회).
- 진급 규칙: 현재 난이도 성공률이 $\text{curr\_up}$ 초과 시 $\text{curr\_inc}$만큼 확대.

> **[Figure 4.3]** 커리큘럼 진행에 따른 목표 분포 확대(근거리/정면 → 원거리/광각).

### 4.3.3 Adaptive Target Entropy ("Entropy Boosting") — **Contribution (C3)**
커리큘럼이 난이도를 올릴 때마다 새 레벨에서 탐색이 필요한데, 고정 target entropy는 $\alpha$가 조기에 붕괴해 새 레벨을 못 배운다. 이를 **적응적 target entropy**로 완화:
$$\mathcal H_{\mathrm{tgt}}=\mathcal H_{\mathrm{base}}+\Delta_{\mathrm{boost}}\cdot\min\big(\underbrace{0.5^{t_{\mathrm{adv}}/T_{1/2}}}_{\text{시간 감쇠(상한)}},\ \underbrace{\tfrac{\text{up}-\text{success}}{\text{up}}}_{\text{성공률 게이트}}\big)$$
- 커리큘럼 상승 직후 탐색을 부스트($+\Delta_{\mathrm{boost}}$), 이후 **성공률↑ 또는 시간경과**에 따라 감쇠 → 필요한 만큼만 탐색.
- `min`으로 두 신호 중 "탐색 줄여" 쪽을 따름(안전) + 시간 상한으로 explore-forever 방지.
- **검증(E3)**: fixed vs adaptive target entropy 비교 → 표본효율·최종 성공률·$\alpha$ 붕괴 곡선. (예비: 성능 향상 확인.)

> **[Figure 4.4]** (a) fixed vs adaptive target entropy 스케줄, (b) policy entropy vs target(붕괴 감지), (c) 성공률 곡선 비교.

### (선택) 그 외 설계선택
- **Start randomization**: 매 reset 관절 랜덤 시작자세 → 상대방위 균일 + stitching(임의 시작→도달의 연쇄)으로 후면·원거리 일반화.
- **Workspace-uniform goal 샘플링**: 도달 envelope 내 부피 균일 + reachability rejection → 원거리 goal 과소표집 교정.

---

## 표기·근거 요약
- $\boldsymbol\Delta_{\max}=[0.12,0.03,0.05,0.08]$, $\Delta t=0.04$s, $\epsilon=0.2$m, $R_{\mathrm{reach}}\approx4.46$m.
- 관측 20-D(obstacle-aware) / 8-D(pure tracker), action 4-D.
- 실제 파일: env=[excavator_env.py](Model/Version4/excavator_env.py), reward=[reward.py](Model/Version4/reward.py), 콜백(curriculum·adaptive entropy·eval)=[callbacks.py](Model/Version4/callbacks.py).

## TODO (작성 시 채울 것)
- [ ] 4.1 Figure(MDP 개요) 제작
- [ ] 4.2 SAC 수식 정리 + HER Figure 제작
- [ ] TTC 계산식은 Sec. 3(Interface)와 중복 안 되게 교차참조
- [ ] 4.3.1/4.3.2 ablation 수치(E4), 4.3.3 ablation 수치(E3) 삽입
- [ ] 가중치·하이퍼파라미터 표 1개로 통합
</content>
