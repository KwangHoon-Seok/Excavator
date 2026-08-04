# RQ2 — Does the VLM mitigate the long-horizon limitations of GCRL?

## 실험 설계 (공정성 확보)
- **환경**: 장애물 **없음**. 순수하게 *목표까지의 거리*(long-horizon) 효과만 분리.
- **목표**: HOME(tip 기준 (0, 3.62, 1.01))에서 거리 0.5–6.9 m로 무작위 샘플된 단일 goal (n=120).
- **평가 변수**: goal의 HOME-거리(=필요한 swing 크기). OOD는 "학습 때 보지 못한 먼 goal"로 **객관적 거리**로 정의 — GCRL이 실패한 것만 골라내는 cherry-picking을 배제.
- **A (단독 GCRL)**: 채택된 컨트롤러(gcrl_v1_startX_dist, 3M)가 goal을 직접 추종.
- **B (VLM-hops + GCRL)**: 동일 컨트롤러. Qwen3.5-27B가 먼 goal을 수평거리 ≤1.5 m의 **짧은 hop 시퀀스**로 분해(마지막 hop=goal) → GCRL이 순차 relay.
- **성공 정의 동일**: tip이 goal에 도달(<{eps}). A·B는 **waypoint 유무만 다르고** 나머지(rollout·성공기준)는 완전히 동일 → 공정 비교.

## 결과
| 구간 | A (GCRL) | B (VLM+GCRL) | Δ |
|------|:---:|:---:|:---:|
| ID (<3.6 m, n=50) | 0.76 | 0.86 | **+0.10** |
| OOD (≥3.6 m, n=70) | 0.01 | 0.60 | **+0.59** |
| **전체 (n=120)** | **0.33** | **0.71** | **+0.38** |

거리 구간별 상세는 `table.csv` / `figure.png` 참조.

## 고찰
1. **근거리(≤2.4 m, ID)에서 A ≈ B.** 단독 GCRL이 이미 신뢰성 있게 도달하는 구간에서는 VLM 분해가 **성능을 해치지 않는다**(중립). 이는 짧은 goal에 불필요한 waypoint를 넣으면 오히려 방해가 된다는 통념과 달리, 마지막 hop을 goal로 고정하고 가까우면 1점만 두도록 프롬프트를 설계해 detour를 억제했기 때문이다.
2. **단독 GCRL의 long-horizon 절벽.** A는 ~3.2 m에서 0.44, ~4.0 m에서 0.05로 급락하고 **4 m 이상에서는 0.00**이다. 커리큘럼이 다룬 유효 반경을 넘어서는 goal은 한 번의 큰 swing으로 도달해야 하는데, 단일 GCRL 정책은 이 long-horizon 제어를 수행하지 못한다.
3. **VLM이 도달 가능 지평(reachable horizon)을 확장.** B는 동일 구간에서 0.33–0.73을 유지한다. 먼 goal을 GCRL이 이미 잘 하는 **짧은 hop들의 연쇄**로 바꿔주면, 각 hop이 학습 분포 안(≤1.5 m)에 들어와 정책이 안정적으로 추종한다. 전체 성공률은 **A=0.33 → B=0.71** (+0.38).
4. **적응적 분해.** VLM이 생성한 hop 수는 거리에 따라 단조 증가(근거리 1.8 → 원거리 4.1, figure (b)). 모델이 goal 거리에 맞춰 분해 세밀도를 스스로 조절함을 보인다.

## 결론
VLM은 GCRL의 long-horizon 한계를 **직접적으로 보완**한다. 단독 GCRL이 완전히 실패하는(성공률 0) 먼 goal 영역에서, VLM의 sub-goal 분해는 도달 성공률을 0 → 0.3–0.7로 끌어올린다. 이 이득은 **거리에서만** 발생하며(근거리에서는 중립), OOD를 *거리*로 객관 정의하고 A·B를 동일 조건에서 평가했으므로 cherry-picking이 아니다.

*(figure: `figure.png` — (a) 거리별 A vs B 성공률, (b) 거리별 hop 수 / 데이터: `data_reach_ab.csv` / 표: `table.csv`,`table.tex`)*
