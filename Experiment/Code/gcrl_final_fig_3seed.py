"""채택 GCRL 모델(start-fixed × dist)의 3-seed mean±std 학습곡선(success/safe-success, 영문).
seed1/2 = Model/logs/GCRL_final/adopted_s{1,2}/fulltask.csv, seed3 = GCRL_compare/gcrl_v1_startX_dist(0-2M)."""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/elicer/Excavator"
LOG = os.path.join(ROOT, "Model", "logs")
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
HORIZON = 2_000_000
GRID = np.arange(10_000, HORIZON + 1, 10_000)


def _read(path):
    if not os.path.exists(path):
        return None
    r = list(csv.DictReader(open(path)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]} if r else None


def seed_series(seed):
    if seed == 3:  # 기존 승자 run: 0-1M 백업 + 1M-3M 본체
        a = _read(os.path.join(LOG, "GCRL_compare", "gcrl_v1_startX_dist", "fulltask_1M.csv"))
        b = _read(os.path.join(LOG, "GCRL_compare", "gcrl_v1_startX_dist", "fulltask.csv"))
        d = {k: np.concatenate([a[k], b[k]]) for k in a} if a and b else (a or b)
    else:
        d = _read(os.path.join(LOG, "GCRL_final", f"adopted_s{seed}", "fulltask.csv"))
    return d


def ema(y, alpha=0.3):
    o = np.copy(y)
    for i in range(1, len(y)):
        o[i] = alpha * y[i] + (1 - alpha) * o[i - 1]
    return o


def stack(metric, seeds=(1, 2, 3)):
    """각 seed를 공통 GRID에 보간 후 [n_seed, len(GRID)] 배열. 없는 seed는 스킵."""
    rows = []
    for s in seeds:
        d = seed_series(s)
        if d is None or metric not in d:
            continue
        m = d["step"] <= HORIZON + 5000
        rows.append(np.interp(GRID, d["step"][m], ema(d[metric][m])))
    return np.array(rows) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 13, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = GRID / 1e6
    for metric, color, lab in [("fulltask_success", "#1f77b4", "Success rate"),
                               ("safe_success", "#2ca02c", "Safe-success rate")]:
        A = stack(metric)
        if A is None:
            continue
        mu, sd = A.mean(0), A.std(0)
        ax.plot(x, mu, "-", color=color, lw=2.4, label=f"{lab} (n={len(A)})")
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.20, linewidth=0)
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Rate")
    ax.set_xlim(0, HORIZON / 1e6); ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc="lower right", framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gcrl_final_3seed.{ext}"), bbox_inches="tight")
    plt.close(fig)
    ns = len(stack("safe_success")) if stack("safe_success") is not None else 0
    print(f"saved gcrl_final_3seed.png  (seeds available={ns})")


if __name__ == "__main__":
    main()
