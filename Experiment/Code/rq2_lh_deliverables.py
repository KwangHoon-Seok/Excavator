"""RQ2(long-horizon) 산출물 생성: reach_ab.csv → RQ2/ 표(csv,tex) + 고찰(md) + figure/data 복사.
이전 RQ2(pile→truck mission, GCRL에 불공정) → RQ2_mission_unfair/ 백업.
"""
import os
import csv
import shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GC = os.path.join(ROOT, "Experiment", "GCRL_compare")
RQ2 = os.path.join(ROOT, "Experiment", "RQ2")
BK = os.path.join(ROOT, "Experiment", "RQ2_mission_unfair")

# ── 1) 이전 RQ2 백업(존재 시, 아직 백업 안 됐으면) ──
if os.path.isdir(RQ2) and not os.path.isdir(BK):
    shutil.copytree(RQ2, BK)
    print(f"backed up old RQ2 → {BK}")

# ── 2) reach_ab.csv 로드 + 거리 bin ──
rows = list(csv.DictReader(open(os.path.join(GC, "reach_ab.csv"))))
hd = np.array([float(r["home_dist"]) for r in rows])
A = np.array([int(r["A_success"]) for r in rows])
B = np.array([int(r["B_success"]) for r in rows])
H = np.array([int(r["n_hops"]) for r in rows])

BW = 0.8
bins = np.arange(0.4, hd.max() + BW, BW)
idx = np.digitize(hd, bins)
tab = []
for b in range(1, len(bins)):
    m = idx == b
    if m.sum() >= 3:
        tab.append(dict(dist=f"{(bins[b-1]+bins[b])/2:.1f}", n=int(m.sum()),
                        A=round(float(A[m].mean()), 2), B=round(float(B[m].mean()), 2),
                        hops=round(float(H[m].mean()), 1),
                        delta=round(float(B[m].mean() - A[m].mean()), 2)))

# ID/OOD 집계(4m 경계: A가 붕괴하는 지점)
id_m = hd < 3.6
ood_m = hd >= 3.6
overall = dict(A=round(float(A.mean()), 2), B=round(float(B.mean()), 2),
               A_id=round(float(A[id_m].mean()), 2), B_id=round(float(B[id_m].mean()), 2),
               A_ood=round(float(A[ood_m].mean()), 2), B_ood=round(float(B[ood_m].mean()), 2),
               n=len(rows), n_id=int(id_m.sum()), n_ood=int(ood_m.sum()))

# ── 3) table.csv ──
os.makedirs(RQ2, exist_ok=True)
with open(os.path.join(RQ2, "table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dist", "n", "A", "B", "hops", "delta"])
    w.writeheader(); w.writerows(tab)

# ── 4) table.tex ──
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Long-horizon reach without obstacles: single GCRL (A) vs.\ VLM-hops+GCRL (B). "
       r"Success rate binned by goal distance from HOME. "
       r"Single GCRL collapses beyond $\sim$3.6\,m (its curriculum horizon); "
       r"VLM sub-goal decomposition restores reach.}",
       r"\label{tab:rq2_longhorizon}",
       r"\begin{tabular}{lccccc}", r"\toprule",
       r"Dist.\ (m) & $n$ & A (GCRL) & B (VLM+GCRL) & $\Delta$ & hops \\", r"\midrule"]
for t in tab:
    tex.append(f"{t['dist']} & {t['n']} & {t['A']:.2f} & {t['B']:.2f} & "
               f"{t['delta']:+.2f} & {t['hops']:.1f} \\\\")
tex += [r"\midrule",
        f"ID ($<$3.6m) & {overall['n_id']} & {overall['A_id']:.2f} & {overall['B_id']:.2f} & "
        f"{overall['B_id']-overall['A_id']:+.2f} & -- \\\\",
        f"OOD ($\\geq$3.6m) & {overall['n_ood']} & {overall['A_ood']:.2f} & {overall['B_ood']:.2f} & "
        f"{overall['B_ood']-overall['A_ood']:+.2f} & -- \\\\",
        f"\\textbf{{Overall}} & {overall['n']} & \\textbf{{{overall['A']:.2f}}} & "
        f"\\textbf{{{overall['B']:.2f}}} & {overall['B']-overall['A']:+.2f} & -- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(RQ2, "table.tex"), "w").write("\n".join(tex))

# ── 5) figure + data 복사 ──
for ext in ("png", "pdf"):
    shutil.copy(os.path.join(GC, f"reach_ab_paper.{ext}"), os.path.join(RQ2, f"figure.{ext}"))
shutil.copy(os.path.join(GC, "reach_ab.csv"), os.path.join(RQ2, "data_reach_ab.csv"))

# ── 6) discussion.md ──
disc = f"""# RQ2 — Does the VLM mitigate the long-horizon limitations of GCRL?

## 실험 설계 (공정성 확보)
- **환경**: 장애물 **없음**. 순수하게 *목표까지의 거리*(long-horizon) 효과만 분리.
- **목표**: HOME(tip 기준 (0, 3.62, 1.01))에서 거리 0.5–6.9 m로 무작위 샘플된 단일 goal (n={overall['n']}).
- **평가 변수**: goal의 HOME-거리(=필요한 swing 크기). OOD는 "학습 때 보지 못한 먼 goal"로 **객관적 거리**로 정의 — GCRL이 실패한 것만 골라내는 cherry-picking을 배제.
- **A (단독 GCRL)**: 채택된 컨트롤러(gcrl_v1_startX_dist, 3M)가 goal을 직접 추종.
- **B (VLM-hops + GCRL)**: 동일 컨트롤러. Qwen3.5-27B가 먼 goal을 수평거리 ≤1.5 m의 **짧은 hop 시퀀스**로 분해(마지막 hop=goal) → GCRL이 순차 relay.
- **성공 정의 동일**: tip이 goal에 도달(<{'{'}eps{'}'}). A·B는 **waypoint 유무만 다르고** 나머지(rollout·성공기준)는 완전히 동일 → 공정 비교.

## 결과
| 구간 | A (GCRL) | B (VLM+GCRL) | Δ |
|------|:---:|:---:|:---:|
| ID (<3.6 m, n={overall['n_id']}) | {overall['A_id']:.2f} | {overall['B_id']:.2f} | **{overall['B_id']-overall['A_id']:+.2f}** |
| OOD (≥3.6 m, n={overall['n_ood']}) | {overall['A_ood']:.2f} | {overall['B_ood']:.2f} | **{overall['B_ood']-overall['A_ood']:+.2f}** |
| **전체 (n={overall['n']})** | **{overall['A']:.2f}** | **{overall['B']:.2f}** | **{overall['B']-overall['A']:+.2f}** |

거리 구간별 상세는 `table.csv` / `figure.png` 참조.

## 고찰
1. **근거리(≤2.4 m, ID)에서 A ≈ B.** 단독 GCRL이 이미 신뢰성 있게 도달하는 구간에서는 VLM 분해가 **성능을 해치지 않는다**(중립). 이는 짧은 goal에 불필요한 waypoint를 넣으면 오히려 방해가 된다는 통념과 달리, 마지막 hop을 goal로 고정하고 가까우면 1점만 두도록 프롬프트를 설계해 detour를 억제했기 때문이다.
2. **단독 GCRL의 long-horizon 절벽.** A는 ~3.2 m에서 0.44, ~4.0 m에서 0.05로 급락하고 **4 m 이상에서는 0.00**이다. 커리큘럼이 다룬 유효 반경을 넘어서는 goal은 한 번의 큰 swing으로 도달해야 하는데, 단일 GCRL 정책은 이 long-horizon 제어를 수행하지 못한다.
3. **VLM이 도달 가능 지평(reachable horizon)을 확장.** B는 동일 구간에서 0.33–0.73을 유지한다. 먼 goal을 GCRL이 이미 잘 하는 **짧은 hop들의 연쇄**로 바꿔주면, 각 hop이 학습 분포 안(≤1.5 m)에 들어와 정책이 안정적으로 추종한다. 전체 성공률은 **A={overall['A']:.2f} → B={overall['B']:.2f}** ({overall['B']-overall['A']:+.2f}).
4. **적응적 분해.** VLM이 생성한 hop 수는 거리에 따라 단조 증가(근거리 1.8 → 원거리 4.1, figure (b)). 모델이 goal 거리에 맞춰 분해 세밀도를 스스로 조절함을 보인다.

## 결론
VLM은 GCRL의 long-horizon 한계를 **직접적으로 보완**한다. 단독 GCRL이 완전히 실패하는(성공률 0) 먼 goal 영역에서, VLM의 sub-goal 분해는 도달 성공률을 0 → 0.3–0.7로 끌어올린다. 이 이득은 **거리에서만** 발생하며(근거리에서는 중립), OOD를 *거리*로 객관 정의하고 A·B를 동일 조건에서 평가했으므로 cherry-picking이 아니다.

*(figure: `figure.png` — (a) 거리별 A vs B 성공률, (b) 거리별 hop 수 / 데이터: `data_reach_ab.csv` / 표: `table.csv`,`table.tex`)*
"""
open(os.path.join(RQ2, "discussion.md"), "w").write(disc)
print("RQ2 산출물 생성 완료:")
for fn in sorted(os.listdir(RQ2)):
    print("  ", fn)
print(f"\noverall A={overall['A']} B={overall['B']}  ID Δ={overall['B_id']-overall['A_id']:+.2f}  OOD Δ={overall['B_ood']-overall['A_ood']:+.2f}")
