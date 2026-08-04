"""Long-horizon A vs B (장애물 없음) — 먼 goal에 대해 단독 GCRL(A) vs VLM-hops+GCRL(B).
A는 reach_vs_dist.csv 재사용(동일 조건). B만 새로: 27B가 goal을 짧은 hop으로 분해 → GCRL relay.
빈 씬은 1회만 렌더(장애물 없어 모든 seed 동일), goal만 바뀜.
출력: Experiment/GCRL_compare/reach_ab.csv + reach_ab_vs_dist.png
"""
import os
import sys
import csv
import json
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np                                              # noqa: E402
import mujoco                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))
sys.path.insert(0, os.path.join(ROOT, "VLM"))
sys.path.insert(0, HERE)
import render_scene                                             # noqa: E402
import interface                                                # noqa: E402
import gridmap                                                  # noqa: E402
from interface import parse_plan                                # noqa: E402
from render_cartesian import render_cart_from_grid             # noqa: E402
from stable_baselines3 import SAC                               # noqa: E402
import qwen_planner                                             # noqa: E402

ADOPTED = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
PROMPT = os.path.join(HERE, "prompts", "reach_hops.txt")
INSTRUCTION = "주어진 목표 좌표에 도달하는 서브골 시퀀스를 계획하라(먼 목표는 짧은 hop으로 분해)."
HOME_TIP = np.array([0.0, 3.62, 1.01])


def reach_rollout(env, model, seed, waypoints, sub_steps, max_steps):
    """장애물 제거 후 waypoints(+goal) relay. 성공 = 궤적 중 tip이 goal 근처(<eps) 도달."""
    env.reset(seed=seed)
    goal = np.asarray(env._desired, float).copy()
    interface.apply_obstacles(env, [])                          # 장애물 제거
    mujoco.mj_forward(env.model, env.data)
    wps = [np.asarray(w, float) for w in waypoints] + [goal.copy()] if waypoints else [goal.copy()]
    obs = env._get_obs(); reached = False
    for wp in wps:
        env._desired = wp; obs = env._get_obs()
        for _ in range(sub_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if float(np.linalg.norm(env._tip() - goal)) < env.eps:
                reached = True
            if float(np.linalg.norm(env._tip() - wp)) < env.eps:
                break
            if trunc:
                break
        if env._steps >= env.max_steps:
            break
    return int(reached)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--qwen_model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--ckpt", default=ADOPTED)
    ap.add_argument("--sub_steps", type=int, default=120)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "reach_hops"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="bf16 적재(기본=4bit, 좀비 GPU 메모리 회피)")
    a = ap.parse_args()
    load_4bit = not a.bf16
    os.makedirs(a.out, exist_ok=True)
    OUTFIG = os.path.join(ROOT, "Experiment", "GCRL_compare")

    env = render_scene.build_env(); env.set_curriculum(1.0); env.max_steps = a.max_steps
    model = SAC.load(a.ckpt, env=env, device="cpu")

    # 빈 씬 1회 렌더(장애물 없음 → 모든 seed 공통): RGB + 빈 Cartesian gridmap
    common = os.path.join(a.out, "_scene")
    env.reset(seed=0); interface.apply_obstacles(env, []); mujoco.mj_forward(env.model, env.data)
    render_scene.render_current(env, common, 0)                 # RGB + polar gridmap + scene.json
    grid = gridmap.polar_height_grid(env.model, env.data, r_max=4.5, object_bodies=[], ground_z=gridmap.GROUND_Z)
    render_cart_from_grid(grid, os.path.join(common, "gridmap.png"))
    rgb_p, grid_p = os.path.join(common, "cabin_init.png"), os.path.join(common, "gridmap.png")

    # A 재사용(reach_vs_dist.csv)
    A_by_seed = {}
    csvA = os.path.join(OUTFIG, "reach_vs_dist.csv")
    if os.path.exists(csvA):
        for r in csv.DictReader(open(csvA)):
            A_by_seed[int(r["seed"])] = int(r["success"])

    rows = []
    for seed in range(a.n):
        env.reset(seed=seed); goal = np.asarray(env._desired, float).copy()
        home_d = float(np.linalg.norm(goal - HOME_TIP))
        # B: VLM hops
        seed_dir = os.path.join(a.out, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        pj = os.path.join(seed_dir, "plan.json")
        if os.path.exists(pj) and not a.refresh:
            hops = parse_plan(pj)["goals"]
        else:
            try:
                qwen_planner.infer(INSTRUCTION, rgb_p, grid_p, model=a.qwen_model, prompt=PROMPT,
                                   goal=goal.tolist(), out=seed_dir, modality="both", load_4bit=load_4bit)
                hops = parse_plan(pj)["goals"]
            except Exception as e:
                open(os.path.join(seed_dir, "err.txt"), "w").write(repr(e) + traceback.format_exc())
                hops = []
        # A와 B는 동일 rollout(성공정의 tip<eps·동일 sub_steps/max_steps)에서 waypoints만 차이 → 공정.
        B = reach_rollout(env, model, seed, hops, a.sub_steps, a.max_steps)
        A = reach_rollout(env, model, seed, [], a.sub_steps, a.max_steps)
        rows.append(dict(seed=seed, home_dist=round(home_d, 3), n_hops=len(hops),
                         A_success=int(A), B_success=int(B)))
        if (seed + 1) % 20 == 0:
            print(f"  {seed+1}/{a.n}  A={np.mean([r['A_success'] for r in rows]):.2f} "
                  f"B={np.mean([r['B_success'] for r in rows]):.2f}", flush=True)

    csvp = os.path.join(OUTFIG, "reach_ab.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    env.close()

    # ── A vs B success-vs-distance 곡선 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hd = np.array([r["home_dist"] for r in rows])
    A = np.array([r["A_success"] for r in rows]); B = np.array([r["B_success"] for r in rows])
    bins = np.arange(0, hd.max() + 0.8, 0.8)
    idx = np.digitize(hd, bins)
    xs, ya, yb, ns = [], [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 3:
            xs.append((bins[b-1]+bins[b])/2); ya.append(A[m].mean()); yb.append(B[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.plot(xs, ya, "-o", color="#888888", lw=2.2, label="Single GCRL (A)")
    ax.plot(xs, yb, "-o", color="#2f7ed8", lw=2.2, label="VLM-hops + GCRL (B)")
    ax.axvspan(5.0, max(xs) + 0.5, color="#ffdddd", alpha=0.4, label="OOD (far)")
    ax.set_xlabel("Goal distance from HOME (m)"); ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=11)
    ax.set_title(f"Long-horizon: VLM decomposition extends reach (no obstacles, n={len(rows)})",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTFIG, f"reach_ab_vs_dist.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n전체 A={A.mean():.2f} B={B.mean():.2f}  → {csvp}, reach_ab_vs_dist.png")
    for x, va, vb, nn in zip(xs, ya, yb, ns):
        print(f"  ~{x:.1f}m: A={va:.2f} B={vb:.2f} (n={nn})")


if __name__ == "__main__":
    main()
