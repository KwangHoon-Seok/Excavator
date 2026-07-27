"""baseline(curriculum·te-boosting OFF) GCRL figure + adopted 대비 비교.
adopted 로그: GCRL_compare/gcrl_v1_startX_dist, GCRL_final/adopted_s{1,2}
baseline 로그: GCRL_baseline/nocurric_s{1,2,3}
출력(Experiment/GCRL_compare/):
  gcrl_baseline_3seed.{png,pdf}            — baseline 3seed mean±std 성능
  gcrl_adopted_vs_baseline.{png,pdf}       — adopted vs baseline 성능 오버레이
  gcrl_baseline_curriculum_entropy.{png,pdf} — baseline 메커니즘(평평: 부스팅/커리큘럼 없음)
--notext 로 텍스트 제거.
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
ADOPTED = {
    "seed 3": os.path.join(ROOT, "Model/logs/GCRL_compare/gcrl_v1_startX_dist/SAC_1"),
    "seed 1": os.path.join(ROOT, "Model/logs/GCRL_final/adopted_s1/SAC_1"),
    "seed 2": os.path.join(ROOT, "Model/logs/GCRL_final/adopted_s2/SAC_1"),
}
BASELINE = {
    "seed 1": os.path.join(ROOT, "Model/logs/GCRL_baseline/nocurric_s1/SAC_1"),
    "seed 2": os.path.join(ROOT, "Model/logs/GCRL_baseline/nocurric_s2/SAC_1"),
    "seed 3": os.path.join(ROOT, "Model/logs/GCRL_baseline/nocurric_s3/SAC_1"),
}
REP_B = "seed 2"


def load(d, tag):
    xs = []
    for ev in sorted(glob.glob(os.path.join(d, "events*"))):
        ea = EventAccumulator(ev, size_guidance={"scalars": 0}); ea.Reload()
        if tag in ea.Tags()["scalars"]:
            xs += [(s.step, s.value) for s in ea.Scalars(tag)]
    xs = sorted(set(xs))
    return (np.array([x[0] for x in xs], float), np.array([x[1] for x in xs], float))


def band(runs, tag, grid):
    M = []
    for d in runs.values():
        st, va = load(d, tag)
        if len(st) < 2:
            continue
        m = (grid >= st.min()) & (grid <= st.max())
        row = np.full(len(grid), np.nan); row[m] = np.interp(grid[m], st, va)
        M.append(row)
    M = np.array(M)
    return np.nanmean(M, 0), np.nanstd(M, 0), M.shape[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--notext", action="store_true")
    T = ap.parse_args().notext
    os.makedirs(OUT, exist_ok=True)
    GRID = np.linspace(1.1e4, 1.99e6, 160)
    suf = "_notext" if T else ""
    SUC, SAFE = "eval_fulltask/success_rate", "eval_fulltask/safe_success_rate"

    # ══ 1) baseline 3-seed 성능 ══
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tag, col, lab in [(SUC, "#2f7ed8", "Success"), (SAFE, "#e2711d", "Safe-success")]:
        mu, sd, n = band(BASELINE, tag, GRID); xm = GRID / 1e6
        ax.plot(xm, mu, "-", color=col, lw=2.4, zorder=3, label=None if T else lab)
        ax.fill_between(xm, mu - sd, mu + sd, color=col, alpha=0.22, zorder=1)
    ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 2.0); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    if not T:
        ax.set_xlabel("Environment steps (millions)", fontsize=12)
        ax.set_ylabel("Evaluation success rate", fontsize=12)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_title("Baseline GCRL — no curriculum, no entropy boosting (3 seeds)", fontsize=11.5, weight="bold")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"gcrl_baseline_3seed{suf}.{e}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══ 2) adopted vs baseline 오버레이 (success + safe) ══
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    xm = GRID / 1e6
    combos = [(ADOPTED, SUC, "#1f4e9c", "-", "Adopted · Success"),
              (BASELINE, SUC, "#7bb0f0", "--", "Baseline · Success"),
              (ADOPTED, SAFE, "#b5530f", "-", "Adopted · Safe-succ"),
              (BASELINE, SAFE, "#f0a97b", "--", "Baseline · Safe-succ")]
    for runs, tag, col, ls, lab in combos:
        mu, sd, n = band(runs, tag, GRID)
        ax.plot(xm, mu, ls, color=col, lw=2.3, zorder=3, label=None if T else lab)
        ax.fill_between(xm, mu - sd, mu + sd, color=col, alpha=0.14, zorder=1)
    ax.set_ylim(-0.02, 1.0); ax.set_xlim(0, 2.0); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    if not T:
        ax.set_xlabel("Environment steps (millions)", fontsize=12)
        ax.set_ylabel("Evaluation success rate", fontsize=12)
        ax.legend(fontsize=9.5, loc="center right", ncol=1, framealpha=0.95)
        ax.set_title("Curriculum + entropy boosting vs. plain SAC+HER (3 seeds each)", fontsize=11.5, weight="bold")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"gcrl_adopted_vs_baseline{suf}.{e}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ══ 3) baseline 메커니즘 패널 (평평: 부스팅/커리큘럼 없음) — adopted와 동일 레이아웃 ══
    d = BASELINE[REP_B]
    te_s, te_v = load(d, "policy/target_entropy")
    ea_s, ea_v = load(d, "policy/entropy_actual")
    al_s, al_v = load(d, "train/ent_coef")
    cr_s, cr_v = load(d, "curriculum/curr_r")   # baseline엔 없을 수 있음
    fig, (axc, axe) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})
    if len(cr_s):
        axc.plot(cr_s / 1e6, cr_v, "-", color="#2e8b57", lw=2.4, drawstyle="steps-post")
        axc.fill_between(cr_s / 1e6, 0, cr_v, color="#2e8b57", alpha=0.12, step="post")
        axc.set_ylim(0, cr_v.max() * 1.12)
    else:   # curriculum 없음 → 처음부터 full-range(고정) 표시
        axc.axhline(8.56, color="#2e8b57", lw=2.4)
        axc.fill_between([0, 2.0], 0, 8.56, color="#2e8b57", alpha=0.08)
        axc.set_ylim(0, 8.56 * 1.12); axc.set_xlim(0, 2.0)
        if not T:
            axc.text(1.0, 4.0, "no curriculum\n(full-range goals from step 0)", ha="center", va="center",
                     fontsize=10, color="#2e8b57")
    axc.grid(alpha=0.3); axc.set_axisbelow(True); axc.tick_params(labelsize=11)
    if not T:
        axc.set_ylabel("Curriculum goal\nradius (m)", fontsize=11.5)
        axc.set_title("Baseline: no curriculum, no target-entropy boosting", fontsize=12, weight="bold")
    axe.plot(te_s / 1e6, te_v, "-", color="#c0392b", lw=2.4, drawstyle="steps-post",
             label=None if T else r"Target entropy $\bar{\mathcal{H}}$ (fixed $-4$)")
    axe.plot(ea_s / 1e6, ea_v, "-", color="#7f8c8d", lw=1.3, alpha=0.8,
             label=None if T else r"Actual policy entropy $\mathcal{H}$")
    axe.axhline(-4.0, ls=":", color="#c0392b", lw=1.0, alpha=0.6)
    axe.set_ylim(min(-5.2, (ea_v.min() * 1.05) if len(ea_v) else -5.2), 2.6); axe.set_xlim(0, 2.0)
    axe.grid(alpha=0.3); axe.set_axisbelow(True); axe.tick_params(labelsize=11)
    axa = axe.twinx()
    axa.plot(al_s / 1e6, al_v, "-", color="#8e44ad", lw=1.6, alpha=0.85, label=None if T else r"Entropy coef. $\alpha$")
    if len(al_v): axa.set_ylim(0, al_v.max() * 1.1)
    axa.tick_params(labelsize=10, colors="#8e44ad")
    if not T:
        axe.set_xlabel("Environment steps (millions)", fontsize=12)
        axe.set_ylabel("Entropy (nats)", fontsize=11.5)
        axa.set_ylabel(r"Entropy coefficient $\alpha$", fontsize=11, color="#8e44ad")
        h1, l1 = axe.get_legend_handles_labels(); h2, l2 = axa.get_legend_handles_labels()
        axe.legend(h1 + h2, l1 + l2, fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"gcrl_baseline_curriculum_entropy{suf}.{e}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 요약
    for name, runs in [("adopted", ADOPTED), ("baseline", BASELINE)]:
        muS, sdS, n = band(runs, SUC, GRID); muF, sdF, _ = band(runs, SAFE, GRID)
        print(f"{name}: n_seed={n} final success={muS[-1]:.3f}±{sdS[-1]:.3f} safe={muF[-1]:.3f}±{sdF[-1]:.3f}")


if __name__ == "__main__":
    main()
