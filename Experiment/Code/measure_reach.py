"""adopted GCRL의 success vs 거리 측정 — 장애물 없이(순수 도달능력) 다수 랜덤 goal.
HOME-거리(swing 크기)·반경 r·방위별 도달 성공률 → OOD(먼 goal) 경계 확인.
출력: Experiment/GCRL_compare/reach_vs_dist.csv + .png
"""
import os
import sys
import csv
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np                                              # noqa: E402
import mujoco                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))  # render_scene
sys.path.insert(0, os.path.join(ROOT, "VLM"))                     # interface
import render_scene                                             # noqa: E402
import interface                                                # noqa: E402
from stable_baselines3 import SAC                               # noqa: E402

ADOPTED = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
HOME_TIP = np.array([0.0, 3.62, 1.01])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="측정 goal 수")
    ap.add_argument("--ckpt", default=ADOPTED)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--obstacles", action="store_true", help="장애물 유지(기본=제거해 순수 거리 효과)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    env = render_scene.build_env(); env.set_curriculum(1.0); env.max_steps = a.max_steps
    model = SAC.load(a.ckpt, env=env, device="cpu")
    rows = []
    for seed in range(a.n):
        env.reset(seed=seed)
        goal = np.asarray(env._desired, float).copy()
        if not a.obstacles:
            interface.apply_obstacles(env, [])                  # 장애물 제거(멀리 park)
            mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        info = {}
        for _ in range(a.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if info.get("is_success") or term or trunc:
                break
        home_d = float(np.linalg.norm(goal - HOME_TIP))
        r = float(np.hypot(goal[0], goal[1]))
        az = float(np.degrees(np.arctan2(goal[0], goal[1])))    # 정면=0, 우+/좌−
        rows.append(dict(seed=seed, home_dist=round(home_d, 3), r=round(r, 3),
                         azimuth=round(az, 1), goal_z=round(float(goal[2]), 3),
                         success=int(bool(info.get("is_success", False))),
                         final_dist=round(float(info.get("dist", np.nan)), 3)))
        if (seed + 1) % 50 == 0:
            sr = np.mean([x["success"] for x in rows])
            print(f"  {seed+1}/{a.n}  running success={sr:.2f}", flush=True)

    csvp = os.path.join(OUT, "reach_vs_dist.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ── success vs HOME-distance (binned) 플롯 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hd = np.array([x["home_dist"] for x in rows]); sc = np.array([x["success"] for x in rows])
    bins = np.arange(0, hd.max() + 0.5, 0.5)
    idx = np.digitize(hd, bins)
    xs, ys, ns = [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 3:
            xs.append((bins[b-1] + bins[b]) / 2); ys.append(sc[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=2.2)
    for x, y, nn in zip(xs, ys, ns):
        ax.annotate(f"{y:.2f}\n(n{nn})", (x, y), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Goal distance from HOME (m)"); ax.set_ylabel("Single-GCRL success rate")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_title(f"Adopted GCRL: success vs goal distance (no obstacles, n={a.n})", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "reach_vs_dist.png"), dpi=140); plt.close(fig)

    print(f"\n전체 success={sc.mean():.2f}  home_dist 범위 [{hd.min():.1f}, {hd.max():.1f}]m  → {csvp}")
    print("거리 구간별 success:")
    for x, y, nn in zip(xs, ys, ns):
        print(f"  ~{x:.1f}m: {y:.2f} (n={nn})")


if __name__ == "__main__":
    main()
