"""
논문용 GCRL 학습 그래프.
(A) 2×2 config 비교(start_fixed/random × dist/auto curriculum) safe_success vs step (→1M) — 채택 근거.
(B) 채택 컨트롤러(v1: start-fixed + dist) 전체 학습곡선 success/safe_success/collision (→3M).
출력: Experiment/GCRL_compare/paper_gcrl.png (+ .pdf)
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/elicer/Excavator"
LOG = os.path.join(ROOT, "Model", "logs", "GCRL_compare")
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")

RUNS = {  # label : (run, color, linestyle)
    "start-fixed × dist  (adopted)": ("gcrl_v1_startX_dist", "#1f77b4", "-"),
    "start-random × dist":           ("gcrl_v2_startO_dist", "#2ca02c", "-"),
    "start-fixed × auto":            ("gcrl_v3_startX_auto", "#ff7f0e", "-"),
    "start-random × auto":           ("gcrl_v4_startO_auto", "#d62728", "-"),
}


def read(run, name="fulltask.csv"):
    p = os.path.join(LOG, run, name)
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    if not r:
        return None
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def ema(y, a=0.35):
    out = np.copy(y)
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    return out


def v1_full():
    """v1: 0-1M(fulltask_1M) + 1-3M(fulltask) 이어붙임."""
    a = read("gcrl_v1_startX_dist", "fulltask_1M.csv")
    b = read("gcrl_v1_startX_dist", "fulltask.csv")
    if a is None:
        return b
    if b is None:
        return a
    return {k: np.concatenate([a[k], b[k]]) for k in a if k in b}


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.axisbelow": True, "figure.dpi": 150})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # ── (A) 2×2 비교 (→1M): safe_success ──
    for label, (run, color, ls) in RUNS.items():
        d = read(run, "fulltask_1M.csv") if run == "gcrl_v1_startX_dist" else read(run)
        if d is None:
            continue
        m = d["step"] <= 1_000_000
        axA.plot(d["step"][m] / 1e6, ema(d["safe_success"][m]), ls, color=color, lw=2.2,
                 label=label + f"  ({d['safe_success'][m][-1]:.2f})")
    axA.set_xlabel("Training steps (M)")
    axA.set_ylabel("Safe success rate")
    axA.set_title("(A) GCRL controller: curriculum × start ablation", fontsize=12, weight="bold")
    axA.set_ylim(0, 0.75); axA.set_xlim(0, 1.0)
    axA.legend(fontsize=9, loc="upper left", framealpha=0.92, title="fixed/random × dist/auto")

    # ── (B) 채택 컨트롤러(v1) 전체 학습곡선 (→3M) ──
    d = v1_full()
    if d is not None:
        st = d["step"] / 1e6
        axB.plot(st, ema(d["fulltask_success"]), "-", color="#1f77b4", lw=2.2, label="success")
        axB.plot(st, ema(d["safe_success"]), "-", color="#2ca02c", lw=2.2, label="safe success (collision-free)")
        axB.plot(st, ema(d["collision_rate"]), "-", color="#d62728", lw=2.0, label="collision rate")
        axB.axvline(1.0, ls="--", color="gray", lw=1.2, alpha=0.8)
        axB.text(1.02, 0.72, "curriculum(1M)\n→ extended", fontsize=8.5, color="gray")
        axB.annotate(f"{d['safe_success'][-1]:.2f}", (st[-1], ema(d['safe_success'])[-1]),
                     fontsize=10, color="#2ca02c", ha="right", va="bottom")
    axB.set_xlabel("Training steps (M)")
    axB.set_ylabel("Rate")
    axB.set_title("(B) Adopted controller (start-fixed × dist)", fontsize=12, weight="bold")
    axB.set_ylim(0, 0.8); axB.legend(fontsize=10, loc="center right", framealpha=0.92)

    fig.suptitle("GCRL low-level controller — training", fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"paper_gcrl.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", os.path.join(OUT, "paper_gcrl.png"), "(+pdf)")


if __name__ == "__main__":
    main()
