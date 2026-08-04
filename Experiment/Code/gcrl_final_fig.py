"""최종 채택 GCRL 모델(v1: start-fixed × dist)의 step별 success / safe-success 곡선(영문, 간결)."""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/elicer/Excavator"
LOG = os.path.join(ROOT, "Model", "logs", "GCRL_compare", "gcrl_v1_startX_dist")
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")


def read(name):
    p = os.path.join(LOG, name)
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]} if r else None


def ema(y, a=0.3):
    o = np.copy(y)
    for i in range(1, len(y)):
        o[i] = a * y[i] + (1 - a) * o[i - 1]
    return o


def main():
    a, b = read("fulltask_1M.csv"), read("fulltask.csv")
    d = {k: np.concatenate([a[k], b[k]]) for k in a} if a and b else (a or b)
    st = d["step"] / 1e6

    plt.rcParams.update({"font.size": 13, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(st, ema(d["fulltask_success"]), "-", color="#1f77b4", lw=2.4, label="Success rate")
    ax.plot(st, ema(d["safe_success"]), "-", color="#2ca02c", lw=2.4, label="Safe-success rate")
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Rate")
    ax.set_xlim(0, st[-1]); ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gcrl_final.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", os.path.join(OUT, "gcrl_final.png"),
          f"(final: success={d['fulltask_success'][-1]:.2f} safe={d['safe_success'][-1]:.2f})")


if __name__ == "__main__":
    main()
