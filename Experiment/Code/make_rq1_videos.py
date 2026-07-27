"""
RQ1 대표 롤아웃 비디오(GIF) 생성 — 모달리티별 셀에서 성공/실패/충돌 seed를 골라
A(단독 GCRL) / B(VLM+GCRL) 롤아웃을 GIF로 저장 → Experiment/RQ1/video/<modality>/.
계획(plan.json)은 셀에서 재사용(VLM 재호출 없음). 눈으로 검증용.

  python Experiment/Code/make_rq1_videos.py
"""
import os
import sys
import csv

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))
sys.path.insert(0, os.path.join(ROOT, "VLM"))

import mission as M                                    # noqa: E402
import mission_eval as ME                              # noqa: E402
from interface import parse_plan                       # noqa: E402
from stable_baselines3 import SAC                      # noqa: E402

CELLS = os.path.join(HERE, "out", "cells")
VIDEO = os.path.join(ROOT, "Experiment", "RQ1", "video")
MODCELL = {
    "both": "Qwen3.5-27B__both__id__rq4_full",
    "rgb_only": "Qwen3.5-27B__rgb_only__id__rq1_rgb_only",
    "height_only": "Qwen3.5-27B__height_only__id__rq1_height_only",
}


def pick_seeds(cell, k_each=2):
    """대표 seed 선정: B성공 / B실패 / (있으면) 충돌. 각 카테고리 최대 k_each개."""
    rows = list(csv.DictReader(open(os.path.join(CELLS, cell, "results.csv"))))
    def I(r, k): return int(float(r[k]))
    succ = [r["seed"] for r in rows if I(r, "B_success")]
    fail = [r["seed"] for r in rows if not I(r, "B_success")]
    coll = [r["seed"] for r in rows if I(r, "B_coll")]
    picked, tag = [], {}
    for s in succ[:k_each]:
        picked.append(s); tag[s] = "success"
    for s in fail[:k_each]:
        if s not in tag:
            picked.append(s); tag[s] = "fail"
    for s in coll[:1]:
        if s not in tag:
            picked.append(s); tag[s] = "collision"
    return picked, tag


def main():
    env = M.build_gcrl_env(); env.set_curriculum(1.0); env.max_steps = 1800
    model = SAC.load(os.path.join(ROOT, "Model", "checkpoints", "Version4", "v4base.zip"),
                     env=env, device="auto")
    rec = ME.Recorder(env.model, h=480, w=640)
    for mod, cell in MODCELL.items():
        cpath = os.path.join(CELLS, cell, "results.csv")
        if not os.path.exists(cpath):
            print(f"[skip] {mod}: {cell} 결과 없음"); continue
        outdir = os.path.join(VIDEO, mod); os.makedirs(outdir, exist_ok=True)
        seeds, tag = pick_seeds(cell)
        print(f"[{mod}] seeds={seeds}")
        for s in seeds:
            ob = M.gen_obstacles(int(s), "id")
            goals = parse_plan(os.path.join(CELLS, cell, f"seed{s}", "plan.json"))["goals"]
            rA = M.rollout_mission(env, model, int(s), ob, [], sub_steps=200, rec=rec)
            rec.save(os.path.join(outdir, f"seed{s}_{tag[s]}_A_gcrl.gif"), stride=3)
            rB = M.rollout_mission(env, model, int(s), ob, goals, sub_steps=200, rec=rec)
            rec.save(os.path.join(outdir, f"seed{s}_{tag[s]}_B_vlm.gif"), stride=3)
            print(f"  seed{s}[{tag[s]}] A(s1={rA['stage1']} s2={rA['stage2']} coll={rA['collided']}) "
                  f"B(s1={rB['stage1']} s2={rB['stage2']} coll={rB['collided']} nsub={rB['n_sub']})")
    rec.close(); env.close()
    print("videos →", VIDEO)


if __name__ == "__main__":
    main()
