"""
Reach+avoid A/B 실험 — 장애물 있는 env에서 **단일 랜덤 goal**에 도달(공정: goal은 GCRL 도달범위 내).
  A (baseline)  : GCRL이 goal로 **직접** 추종(장애물 만나면 local 회피/후퇴)
  B (VLM-guided): 27B가 goal+장애물 보고 **회피 waypoint** 생성 → GCRL이 relay
env = 장애물 학습 env(흙더미·트럭 없음), 단일 goal. gridmap = Cartesian. GCRL = adopted.

  python Experiment/Code/reach_run.py --seeds_file Experiment/Code/reach_seeds.json
"""
import os
import sys
import csv
import json
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))  # render_scene, mission_eval
sys.path.insert(0, os.path.join(ROOT, "VLM"))                     # gridmap, interface
sys.path.insert(0, HERE)                                          # qwen_planner, render_cartesian

import render_scene                                              # noqa: E402
import mission_eval as ME                                        # noqa: E402
import gridmap                                                   # noqa: E402
from interface import parse_plan                                 # noqa: E402
from render_cartesian import render_cart_from_grid               # noqa: E402
from stable_baselines3 import SAC                                # noqa: E402
import qwen_planner                                              # noqa: E402

ADOPTED = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
PROMPT = os.path.join(HERE, "prompts", "reach_cartesian.txt")
INSTRUCTION = "주어진 목표 좌표에 장애물을 피해 도달하는 서브골 시퀀스를 계획하라."
HOME_XY = np.array([0.0, 3.62])


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def is_blocked(env, seed, corridor=0.5, min_goal_dist=1.8):
    """seed reset 후 HOME→goal 직선에 장애물이 있고 goal이 충분히 멀면 challenge."""
    env.reset(seed=seed)
    goal = np.asarray(env._desired[:2], float)
    if float(np.linalg.norm(goal - HOME_XY)) < min_goal_dist:
        return False
    for o in render_scene.active_obstacles(env):
        c = np.array(o["center"][:2]); rad = max(o["half"][:2])
        if _seg_dist(c, HOME_XY, goal) < corridor + rad:
            return True
    return False


def render_cartesian_scene(env, seed_dir, seed):
    """render_scene.render_current(RGB+scene.json) 후 gridmap.png를 Cartesian으로 덮어씀."""
    scene = render_scene.render_current(env, seed_dir, seed)     # RGB + polar gridmap + scene.json
    grid = gridmap.polar_height_grid(env.model, env.data, r_max=4.5,
                                     object_bodies=[o["body"] for o in scene["obstacles"]],
                                     ground_z=gridmap.GROUND_Z)
    render_cart_from_grid(grid, os.path.join(seed_dir, "gridmap.png"))   # VLM 입력=Cartesian
    return scene


def plan_one(seed_dir, goal, model_id, refresh):
    pj = os.path.join(seed_dir, "plan.json")
    if os.path.exists(pj) and not refresh:
        return parse_plan(pj)["goals"], "cached"
    try:
        qwen_planner.infer(INSTRUCTION, os.path.join(seed_dir, "cabin_init.png"),
                           os.path.join(seed_dir, "gridmap.png"),
                           model=model_id, prompt=PROMPT, goal=goal, out=seed_dir, modality="both")
        return parse_plan(pj)["goals"], "planned"
    except Exception as e:
        open(os.path.join(seed_dir, "vlm_error.txt"), "w").write(repr(e) + "\n" + traceback.format_exc())
        return [], f"FAILED({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--ckpt", default=ADOPTED)
    ap.add_argument("--seeds_file", default=None)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--sub_steps", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "reach", "adopted_27B"))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    env = render_scene.build_env(); env.set_curriculum(1.0); env.max_steps = a.max_steps
    model = SAC.load(a.ckpt, env=env, device="auto")
    os.makedirs(a.out, exist_ok=True)
    if a.seeds_file:
        seeds = [int(s) for s in json.load(open(a.seeds_file))]
    else:
        seeds = list(range(a.seed0, a.seed0 + a.n_seeds))
    print(f"reach+avoid: ckpt={os.path.basename(a.ckpt)} qwen={a.qwen_model} seeds={len(seeds)} → {a.out}")

    rows = []
    for s in seeds:
        seed_dir = os.path.join(a.out, f"seed{s}")
        env.reset(seed=s)
        scene = render_cartesian_scene(env, seed_dir, s)
        goals, status = plan_one(seed_dir, scene["goal"], a.qwen_model, a.refresh)
        rA = ME.rollout_A(env, model, s)
        rB = ME.rollout_B(env, model, s, goals, a.sub_steps)
        rows.append(dict(seed=s, model=a.qwen_model.split("/")[-1], n_obs=len(scene["obstacles"]),
                         n_sub=len(goals), goal_dist=round(float(np.linalg.norm(np.array(scene["goal"][:2]) - HOME_XY)), 2),
                         A_success=int(rA["success"]), A_coll=int(rA["collided"]), A_safe=int(rA["safe"]),
                         B_success=int(rB["success"]), B_coll=int(rB["collided"]), B_safe=int(rB["safe"])))
        print(f"  seed {s}: plan={status} sub={len(goals)} obs={len(scene['obstacles'])} "
              f"A(succ={rA['success']:d} safe={rA['safe']:d} coll={rA['collided']:d}) "
              f"B(succ={rB['success']:d} safe={rB['safe']:d} coll={rB['collided']:d})")
    csvp = os.path.join(a.out, "results.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    env.close()
    n = len(rows); rate = lambda k: np.mean([r[k] for r in rows])
    print(f"[reach] n={n}  A_success={rate('A_success'):.2f} B_success={rate('B_success'):.2f}  "
          f"A_safe={rate('A_safe'):.2f} B_safe={rate('B_safe'):.2f}  saved {csvp}")


if __name__ == "__main__":
    main()
