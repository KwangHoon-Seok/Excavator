"""RQ2 (long-horizon) 집계 — 단독 GCRL(A) vs VLM+GCRL(B) success rate, ID.
입력: out/rq2_lh/adopted_27B/results.csv (adopted GCRL + Qwen3.5-27B, 100 ID seeds).
출력: Experiment/RQ2/  table.csv · table.tex · figure.png · discussion.md
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(HERE, "out", "rq2_lh", "adopted_27B", "results.csv")
OUT = os.path.join(ROOT, "Experiment", "RQ2")


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0
    return float(z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / (1 + z * z / n))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(SRC)))
    n = len(rows)
    f = lambda k: float(np.mean([float(r[k]) for r in rows]))
    # metric: (label, A_key, B_key)
    METR = [("Success", "A_success", "B_success"),
            ("Safe-success", "A_safe", "B_safe"),
            ("Pile reach (stage1)", "A_stage1", "B_stage1"),
            ("Truck reach (stage2)", "A_stage2", "B_stage2"),
            ("Collision", "A_coll", "B_coll")]
    A = [f(a) for _, a, _ in METR]
    B = [f(b) for _, _, b in METR]
    eA = [wilson(v, n) for v in A]
    eB = [wilson(v, n) for v in B]
    cats = [m[0] for m in METR]

    # ── figure: A vs B grouped bars ──
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, A, w, yerr=eA, capsize=3, color="#888888", label="Single GCRL (A)", edgecolor="white")
    ax.bar(x + w / 2, B, w, yerr=eB, capsize=3, color="#2f7ed8", label="VLM + GCRL (B, ours)", edgecolor="white")
    for xi, v in zip(x - w / 2, A):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, B):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.set_title(f"RQ2: VLM mitigates GCRL long-horizon limitation (ID, n={n})", fontsize=12, weight="bold")
    ax.legend(fontsize=11, loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"figure.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ── table (CSV + LaTeX) ──
    hdr = ["metric", "Single GCRL (A)", "VLM+GCRL (B)", "Δ(B−A)"]
    trows = [[m[0], f"{a:.3f}", f"{b:.3f}", f"{b-a:+.3f}"] for m, a, b in zip(METR, A, B)]
    with open(os.path.join(OUT, "table.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh); w_.writerow(hdr); w_.writerows(trows)
    with open(os.path.join(OUT, "table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrr}\n\\toprule\n" + " & ".join(hdr) + " \\\\\n\\midrule\n")
        for r in trows:
            fh.write(" & ".join(r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # 조건별 데이터 원본 복사
    import shutil
    shutil.copy(SRC, os.path.join(OUT, "data_adopted_27B.csv"))

    # ── discussion ──
    a_s, b_s = f("A_success"), f("B_success")
    a_s1, a_s2 = f("A_stage1"), f("A_stage2")
    disc = f"""# RQ2 — Does the VLM mitigate the long-horizon limitations of GCRL?

**설정**: ID 환경(고정 레이아웃), n={n} seeds. 저수준 컨트롤러 = adopted GCRL(start-fixed × dist),
플래너 = Qwen3.5-27B. A = 단독 GCRL이 [흙더미, 트럭] 직접 추종 / B = VLM 서브골 relay.

## 결과 (success rate)
| metric | Single GCRL (A) | VLM+GCRL (B) | Δ |
|---|---|---|---|
""" + "".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n" for r in trows) + f"""
## 고찰
- **단독 GCRL success {a_s:.2f} ≪ VLM+GCRL {b_s:.2f}** — VLM 서브골이 long-horizon 임무 성공률을 크게 끌어올림.
- 단계 분해가 핵심 증거: 단독 GCRL은 **흙더미 도달 {a_s1:.2f}** 은 되지만 **트럭 도달 {a_s2:.2f}** 에서 막힘.
  즉 컨트롤러가 흙더미(중간 swing)엔 가지만 흙더미→트럭의 **큰 swing(far goal)** 을 한 번에 수행 못 함.
- VLM은 이 long-horizon 문제를 **짧은 서브골 hop**으로 쪼개어(trucك 도달 {f("B_stage2"):.2f}), GCRL이 각 hop을
  순차 추종하도록 만들어 임무를 완수시킴. → "VLM sub-goal 생성이 long-horizon 문제를 잘게 쪼개 해결 가능한
  문제로 바꿔준다"는 주장을 정량적으로 뒷받침.
- 평가를 OOD가 아닌 **ID에서** 수행해, 분포 변화가 아니라 순수 long-horizon 능력만 격리해 측정.
"""
    open(os.path.join(OUT, "discussion.md"), "w", encoding="utf-8").write(disc)
    print(f"RQ2(long-horizon) → {OUT}  (A_success={a_s:.2f} vs B_success={b_s:.2f}, n={n})")


if __name__ == "__main__":
    main()
