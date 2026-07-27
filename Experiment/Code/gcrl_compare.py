"""
GCRL 4버전(2×2: random_start × curriculum) 학습 성능 실시간 비교·저장.
입력: Model/logs/GCRL_compare/<run>/fulltask.csv (step, fulltask_success, mean_dist, safe_success, collision_rate)
출력: Experiment/GCRL_compare/  progress.png · latest.csv · summary.md

  python Experiment/Code/gcrl_compare.py            # 1회
  watch: 아래 while 루프(train 중 주기 갱신)
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/elicer/Excavator"
LOGDIR = os.path.join(ROOT, "Model", "logs", "GCRL_compare")
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
RUNS = {
    "v1 startX·dist": "gcrl_v1_startX_dist",
    "v2 startO·dist": "gcrl_v2_startO_dist",
    "v3 startX·auto": "gcrl_v3_startX_auto",
    "v4 startO·auto": "gcrl_v4_startO_auto",
}
COLORS = {"v1 startX·dist": "#1f77b4", "v2 startO·dist": "#2ca02c",
          "v3 startX·auto": "#ff7f0e", "v4 startO·auto": "#d62728"}


def load(run):
    p = os.path.join(LOGDIR, run, "fulltask.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    arr = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return arr


def read_tb(run, tags):
    """run의 TB event에서 지정 scalar 태그들 추출 → {tag:(steps,vals)}. 없으면 빈 dict."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rundir = os.path.join(LOGDIR, run)
    # event 파일은 <run>/SAC_1/ 안에 있음
    subs = [os.path.join(rundir, d) for d in os.listdir(rundir)] if os.path.isdir(rundir) else []
    evdir = next((s for s in subs if os.path.isdir(s) and any("tfevents" in f for f in os.listdir(s))), None)
    if evdir is None:
        return {}
    try:
        ea = EventAccumulator(evdir, size_guidance={"scalars": 0}); ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        out = {}
        for t in tags:
            if t in avail:
                ev = ea.Scalars(t)
                out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
        return out
    except Exception:
        return {}


def do_curriculum():
    """curriculum 진행도 저장: dist=frac·target_entropy / auto=mastered·frontier·bin_success."""
    dist_runs = {"v1 startX·dist": "gcrl_v1_startX_dist", "v2 startO·dist": "gcrl_v2_startO_dist"}
    auto_runs = {"v3 startX·auto": "gcrl_v3_startX_auto", "v4 startO·auto": "gcrl_v4_startO_auto"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (1) 커리큘럼 진행: dist frac(실선) + auto mastered_frac(파선) — 둘 다 "열린/숙달된 공간 비율"
    for label, run in dist_runs.items():
        d = read_tb(run, ["curriculum/frac"])
        if "curriculum/frac" in d:
            s, v = d["curriculum/frac"]; ax[0].plot(s / 1000, v, "-", color=COLORS[label], label=label + " (frac)")
    for label, run in auto_runs.items():
        d = read_tb(run, ["autocurr/mastered_frac"])
        if "autocurr/mastered_frac" in d:
            s, v = d["autocurr/mastered_frac"]; ax[0].plot(s / 1000, v, "--", color=COLORS[label], label=label + " (mastered)")
    ax[0].set_title("커리큘럼 진행 (dist=frac / auto=mastered_frac)", fontsize=10, weight="bold")
    ax[0].set_ylim(-0.02, 1.05); ax[0].legend(fontsize=7); ax[0].set_xlabel("step (k)")
    # (2) target_entropy (dist 적응 엔트로피)
    for label, run in dist_runs.items():
        d = read_tb(run, ["curriculum/target_entropy"])
        if "curriculum/target_entropy" in d:
            s, v = d["curriculum/target_entropy"]; ax[1].plot(s / 1000, v, "-", color=COLORS[label], label=label)
    ax[1].set_title("target_entropy (dist 적응)", fontsize=10, weight="bold")
    ax[1].legend(fontsize=7); ax[1].set_xlabel("step (k)"); ax[1].grid(alpha=0.3)
    # (3) auto bin 성공률(frontier/mean)
    for label, run in auto_runs.items():
        d = read_tb(run, ["autocurr/bin_success_mean", "autocurr/frontier_frac"])
        if "autocurr/bin_success_mean" in d:
            s, v = d["autocurr/bin_success_mean"]; ax[2].plot(s / 1000, v, "-", color=COLORS[label], label=label + " bin_succ")
        if "autocurr/frontier_frac" in d:
            s, v = d["autocurr/frontier_frac"]; ax[2].plot(s / 1000, v, ":", color=COLORS[label], label=label + " frontier")
    ax[2].set_title("auto curriculum bin 상태", fontsize=10, weight="bold")
    ax[2].set_ylim(-0.02, 1.05); ax[2].legend(fontsize=7); ax[2].set_xlabel("step (k)"); ax[2].grid(alpha=0.3)
    fig.suptitle("GCRL 커리큘럼 진행 (실시간)", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "curriculum.png"), dpi=140); plt.close(fig)
    # 스냅샷 CSV
    with open(os.path.join(OUT, "curriculum_latest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["version", "step", "dist_frac", "target_entropy", "auto_mastered", "auto_frontier", "auto_bin_succ"])
        for label, run in {**dist_runs, **auto_runs}.items():
            d = read_tb(run, ["curriculum/frac", "curriculum/target_entropy", "autocurr/mastered_frac",
                              "autocurr/frontier_frac", "autocurr/bin_success_mean"])
            g = lambda t: f"{d[t][1][-1]:.3f}" if t in d and len(d[t][1]) else "-"
            step = max((d[t][0][-1] for t in d if len(d[t][0])), default="-")
            w.writerow([label, step, g("curriculum/frac"), g("curriculum/target_entropy"),
                        g("autocurr/mastered_frac"), g("autocurr/frontier_frac"), g("autocurr/bin_success_mean")])


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {label: load(run) for label, run in RUNS.items()}
    have = {k: v for k, v in data.items() if v is not None}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    metrics = [("fulltask_success", "Full-task success"),
               ("safe_success", "Safe success (무충돌 완주)"),
               ("collision_rate", "Collision rate")]
    for ax, (key, title) in zip(axes, metrics):
        for label, arr in have.items():
            if key in arr:
                ax.plot(arr["step"] / 1000, arr[key], "-", color=COLORS[label], label=label, lw=1.8)
        ax.set_xlabel("step (k)"); ax.set_ylabel("rate"); ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("GCRL 2×2 학습 비교 (random_start × curriculum)", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "progress.png"), dpi=140); plt.close(fig)

    # 최신 스냅샷 테이블
    hdr = ["version", "step", "fulltask_success", "safe_success", "collision_rate", "mean_dist"]
    trows = []
    for label in RUNS:
        a = data.get(label)
        if a is None:
            trows.append([label, "-", "-", "-", "-", "-"]); continue
        i = -1
        trows.append([label, int(a["step"][i]),
                      f"{a.get('fulltask_success',[np.nan])[i]:.3f}",
                      f"{a.get('safe_success',[np.nan])[i]:.3f}",
                      f"{a.get('collision_rate',[np.nan])[i]:.3f}",
                      f"{a.get('fulltask_mean_dist',[np.nan])[i]:.3f}"])
    with open(os.path.join(OUT, "latest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(trows)

    # summary.md
    best = None
    for label in RUNS:
        a = data.get(label)
        if a is not None and (best is None or a.get("safe_success", [0])[-1] > best[1]):
            best = (label, a.get("safe_success", [0])[-1], int(a["step"][-1]))
    md = "# GCRL 2×2 학습 실시간 비교\n\n최신 스냅샷 (fulltask 평가, n_eval_full=50):\n\n"
    md += "| version | step | success | safe_success | collision |\n|---|---|---|---|---|\n"
    for r in trows:
        md += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"
    if best:
        md += f"\n**현재 선두**: {best[0]} (safe_success={best[1]:.3f} @ {best[2]} step)\n"
    md += "\n※ 학습 진행 중 자동 갱신. 곡선은 progress.png.\n"
    open(os.path.join(OUT, "summary.md"), "w").write(md)
    try:
        do_curriculum()
    except Exception as e:
        print("  (curriculum 저장 스킵:", e, ")")
    print(f"갱신: {len(have)}/4 버전 데이터  → {OUT}")
    for r in trows:
        print("  ", r[0], "step", r[1], "safe", r[3])


if __name__ == "__main__":
    main()
