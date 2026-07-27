"""reach+avoid challenge seed 선별 — HOME→goal 직선에 장애물이 있고 goal이 충분히 먼 seed 100개.
결과 reach_seeds.json."""
import os
import sys
import json
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))
sys.path.insert(0, HERE)
import render_scene                                              # noqa: E402
import reach_run                                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=100)
    ap.add_argument("--scan", type=int, default=2000)
    ap.add_argument("--corridor", type=float, default=0.5)
    ap.add_argument("--min_goal_dist", type=float, default=1.8)
    ap.add_argument("--out", default=os.path.join(HERE, "reach_seeds.json"))
    a = ap.parse_args()
    env = render_scene.build_env(); env.set_curriculum(1.0)
    seeds = []
    for s in range(a.scan):
        if reach_run.is_blocked(env, s, a.corridor, a.min_goal_dist):
            seeds.append(s)
        if len(seeds) >= a.need:
            break
    json.dump(seeds, open(a.out, "w"))
    print(f"채택 {len(seeds)}/{a.need} (스캔 {s+1})  → {a.out}\nseeds[:15]={seeds[:15]}")


if __name__ == "__main__":
    main()
