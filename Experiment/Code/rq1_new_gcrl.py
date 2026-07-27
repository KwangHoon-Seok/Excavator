"""
RQ1 재측정 — 새로 학습된 GCRL 컨트롤러(gcrl_v1_startX_dist, 3M step)로 롤아웃만 다시 실행.
VLM 계획(plan.json)은 GCRL과 무관 → 기존 27B 셀에서 재사용(VLM 재추론 없음). A/B 롤아웃만 신규.

입력: out/cells/<27B 셀>/seed*/plan.json  +  새 CKPT
출력: out/cells_<gcrl>/<셀>/results.csv  +  Experiment/RQ1/<gcrl>/ (table/figure/discussion)

  python Experiment/Code/rq1_new_gcrl.py
  python Experiment/Code/rq1_new_gcrl.py --ckpt <zip> --tag <name>
"""
import os
import sys
import csv
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))
sys.path.insert(0, os.path.join(ROOT, "VLM"))

import numpy as np                                         # noqa: E402
import mission as M                                        # noqa: E402
from interface import parse_plan                           # noqa: E402
from stable_baselines3 import SAC                          # noqa: E402
import aggregate_rqs as AG                                 # noqa: E402  (summ/wilson/bars 재사용)

RQ1_CELLS = {
    "both": "Qwen3.5-27B__both__id__rq4_full",
    "rgb_only": "Qwen3.5-27B__rgb_only__id__rq1_rgb_only",
    "height_only": "Qwen3.5-27B__height_only__id__rq1_height_only",
}
FIELDS = ["seed", "dist", "modality", "model", "prompt", "n_obs", "n_sub",
          "A_success", "A_coll", "A_safe", "A_stage1", "A_stage2", "A_path",
          "B_success", "B_coll", "B_safe", "B_stage1", "B_stage2", "B_path"]


def run_cell(env, model, src_cell, out_cell, sub_steps=200):
    """src_cell의 plan.json 재사용, 새 컨트롤러로 A/B 롤아웃 → out_cell/results.csv."""
    os.makedirs(out_cell, exist_ok=True)
    seed_dirs = sorted((d for d in os.listdir(src_cell) if d.startswith("seed")),
                       key=lambda d: int(d[4:]))
    rows = []
    for d in seed_dirs:
        seed = int(d[4:])
        pj = os.path.join(src_cell, d, "plan.json")
        if not os.path.exists(pj):
            continue
        p = parse_plan(pj)
        goals = p["goals"]
        ob = M.gen_obstacles(seed, "id")
        rA = M.rollout_mission(env, model, seed, ob, [], sub_steps=sub_steps)
        rB = M.rollout_mission(env, model, seed, ob, goals, sub_steps=sub_steps)
        rows.append(dict(seed=seed, dist="id", modality=os.path.basename(out_cell),
                         model="Qwen3.5-27B", prompt="", n_obs=len(ob), n_sub=rB["n_sub"],
                         A_success=rA["success"], A_coll=rA["collided"], A_safe=rA["safe"],
                         A_stage1=rA["stage1"], A_stage2=rA["stage2"], A_path=rA["path"],
                         B_success=rB["success"], B_coll=rB["collided"], B_safe=rB["safe"],
                         B_stage1=rB["stage1"], B_stage2=rB["stage2"], B_path=rB["path"]))
    with open(os.path.join(out_cell, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "Model", "checkpoints",
                                                   "GCRL_compare", "gcrl_v1_startX_dist.zip"))
    ap.add_argument("--tag", default="gcrl_v1_startX_dist")
    ap.add_argument("--sub_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=1800)
    a = ap.parse_args()

    cells_root = os.path.join(HERE, "out", "cells")
    out_root = os.path.join(HERE, "out", f"cells_{a.tag}")
    env = M.build_gcrl_env(); env.set_curriculum(1.0); env.max_steps = a.max_steps
    model = SAC.load(a.ckpt, env=env, device="auto")
    print(f"[rq1_new_gcrl] ckpt={os.path.basename(a.ckpt)} tag={a.tag}")

    cell_map = {}
    for mod, cid in RQ1_CELLS.items():
        src = os.path.join(cells_root, cid)
        if not os.path.exists(src):
            print(f"[skip] {mod}: src 셀 없음 {cid}"); continue
        out_cell = os.path.join(out_root, cid)
        rows = run_cell(env, model, src, out_cell, a.sub_steps)
        cell_map[mod] = cid
        sB = np.mean([r["B_safe"] for r in rows]); sS = np.mean([r["B_success"] for r in rows])
        cB = np.mean([r["B_coll"] for r in rows])
        print(f"  {mod}: n={len(rows)} B_safe={sB:.3f} B_succ={sS:.3f} B_coll={cB:.3f}")
    env.close()

    # 집계 — aggregate_rqs 재사용(CELLS를 새 경로로 가리키게)
    AG.CELLS = out_root
    outdir = os.path.join(ROOT, "Experiment", "RQ1", a.tag)
    os.makedirs(outdir, exist_ok=True)
    rq = {"RQ1": cell_map}
    msg = AG.do_rq1(rq, outdir)
    # discussion에 GCRL 컨트롤러 명시(헤더 보강)
    dp = os.path.join(outdir, "discussion.md")
    if os.path.exists(dp):
        txt = open(dp).read().replace(
            "# RQ1 — 멀티모달 입력의 영향성",
            f"# RQ1 — 멀티모달 입력의 영향성  (GCRL 컨트롤러: {a.tag}, 3M step)")
        open(dp, "w").write(txt)
    print(msg, "→", outdir)


if __name__ == "__main__":
    main()
