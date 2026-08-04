"""채택 GCRL 학습 결과 + curriculum/target-entropy 메커니즘 논문 figure.
로그: tensorboard (GCRL_compare/gcrl_v1_startX_dist, GCRL_final/adopted_s1, adopted_s2).
출력(Experiment/GCRL_compare/):
  gcrl_train_3seed.{png,pdf}       — 3 seed mean±std success/safe-success vs steps
  gcrl_curriculum_entropy.{png,pdf} — curriculum 반경 + target/actual entropy + alpha (대표 seed)
--notext 로 텍스트 제거 버전.
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
RUNS = {  # seed 라벨 -> 로그 디렉토리
    "seed 3": os.path.join(ROOT, "Model/logs/GCRL_compare/gcrl_v1_startX_dist/SAC_1"),
    "seed 1": os.path.join(ROOT, "Model/logs/GCRL_final/adopted_s1/SAC_1"),
    "seed 2": os.path.join(ROOT, "Model/logs/GCRL_final/adopted_s2/SAC_1"),
}
REP = "seed 2"  # 메커니즘용 대표(깨끗한 단일 run, curriculum 완주 + boost 2회)


def load(d, tag):
    xs = []
    for ev in sorted(glob.glob(os.path.join(d, "events*"))):
        ea = EventAccumulator(ev, size_guidance={"scalars": 0}); ea.Reload()
        if tag in ea.Tags()["scalars"]:
            xs += [(s.step, s.value) for s in ea.Scalars(tag)]
    xs = sorted(set(xs))
    return (np.array([x[0] for x in xs], float), np.array([x[1] for x in xs], float))


def interp_grid(runs, tag, grid):
    """각 seed를 공통 step grid에 보간 → (n_seed, len(grid)) 배열."""
    M = []
    for d in runs.values():
        st, va = load(d, tag)
        if len(st) < 2:
            continue
        m = (grid >= st.min()) & (grid <= st.max())
        row = np.full(len(grid), np.nan)
        row[m] = np.interp(grid[m], st, va)
        M.append(row)
    return np.array(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notext", action="store_true")
    a = ap.parse_args()
    T = a.notext
    os.makedirs(OUT, exist_ok=True)

    # ══ Figure 1: 3-seed 학습 성능 (mean±std) ══
    GRID = np.linspace(1.1e4, 1.99e6, 160)   # 공통 구간 0.011M~1.99M (세 seed 모두 커버)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tag, col, lab in [("eval_fulltask/success_rate", "#2f7ed8", "Success"),
                          ("eval_fulltask/safe_success_rate", "#e2711d", "Safe-success")]:
        M = interp_grid(RUNS, tag, GRID)
        mu = np.nanmean(M, 0); sd = np.nanstd(M, 0)
        xm = GRID / 1e6
        ax.plot(xm, mu, "-", color=col, lw=2.4, zorder=3, label=None if T else lab)
        ax.fill_between(xm, mu - sd, mu + sd, color=col, alpha=0.22, zorder=1)
    ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 2.0)
    ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.tick_params(labelsize=11)
    if not T:
        ax.set_xlabel("Environment steps (millions)", fontsize=12)
        ax.set_ylabel("Evaluation success rate", fontsize=12)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_title("Adopted GCRL training (3 seeds, mean ± std)", fontsize=12, weight="bold")
    fig.tight_layout()
    suf = "_notext" if T else ""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gcrl_train_3seed{suf}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══ Figure 2: curriculum + target-entropy 결합 메커니즘 (대표 seed) ══
    d = RUNS[REP]
    cr_s, cr_v = load(d, "curriculum/curr_r")
    te_s, te_v = load(d, "policy/target_entropy")
    ea_s, ea_v = load(d, "policy/entropy_actual")
    al_s, al_v = load(d, "train/ent_coef")

    fig, (axc, axe) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})
    # (상) curriculum 반경
    axc.plot(cr_s / 1e6, cr_v, "-", color="#2e8b57", lw=2.4, drawstyle="steps-post")
    axc.fill_between(cr_s / 1e6, 0, cr_v, color="#2e8b57", alpha=0.12, step="post")
    axc.set_ylim(0, cr_v.max() * 1.12)
    axc.grid(alpha=0.3); axc.set_axisbelow(True); axc.tick_params(labelsize=11)
    if not T:
        axc.set_ylabel("Curriculum goal\nradius (m)", fontsize=11.5)
        axc.set_title("Curriculum learning drives target-entropy boosting", fontsize=12, weight="bold")

    # (하) target/actual entropy (좌) + alpha (우)
    axe.plot(te_s / 1e6, te_v, "-", color="#c0392b", lw=2.4, drawstyle="steps-post",
             label=None if T else r"Target entropy $\bar{\mathcal{H}}$ (boosted schedule)")
    axe.plot(ea_s / 1e6, ea_v, "-", color="#7f8c8d", lw=1.3, alpha=0.8,
             label=None if T else r"Actual policy entropy $\mathcal{H}$")
    axe.axhline(-4.0, ls=":", color="#c0392b", lw=1.0, alpha=0.6)
    if not T:
        axe.annotate("default $-\\dim(a)=-4$", (0.02, -4.0), fontsize=8.5, color="#c0392b",
                     va="bottom", ha="left")
    axe.set_ylim(min(-5.2, ea_v.min() * 1.05), 2.6)
    axe.grid(alpha=0.3); axe.set_axisbelow(True); axe.tick_params(labelsize=11)
    axa = axe.twinx()
    axa.plot(al_s / 1e6, al_v, "-", color="#8e44ad", lw=1.6, alpha=0.85,
             label=None if T else r"Entropy coef. $\alpha$")
    axa.set_ylim(0, al_v.max() * 1.1); axa.tick_params(labelsize=10, colors="#8e44ad")
    if not T:
        axe.set_xlabel("Environment steps (millions)", fontsize=12)
        axe.set_ylabel("Entropy (nats)", fontsize=11.5)
        axa.set_ylabel(r"Entropy coefficient $\alpha$", fontsize=11, color="#8e44ad")
        h1, l1 = axe.get_legend_handles_labels(); h2, l2 = axa.get_legend_handles_labels()
        axe.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gcrl_curriculum_entropy{suf}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 요약 출력
    print(f"saved gcrl_train_3seed{suf}, gcrl_curriculum_entropy{suf} (rep={REP}, notext={T})")
    for tag in ["eval_fulltask/success_rate", "eval_fulltask/safe_success_rate"]:
        M = interp_grid(RUNS, tag, GRID)
        print(f"  {tag}: final mean={np.nanmean(M[:, -1]):.3f} std={np.nanstd(M[:, -1]):.3f}")


if __name__ == "__main__":
    main()
