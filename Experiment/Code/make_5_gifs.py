"""VLM(27B, both) 계획을 GCRL v1_startX_dist가 추종하는 B 롤아웃 GIF 5개 저장.
→ Experiment/RQ1/v1_newprompt/gifs/seed<N>_B_vlm.gif
계획은 out/cells_v1_newprompt/both 셀에서 재사용(VLM 재호출 없음)."""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))
sys.path.insert(0, os.path.join(ROOT, "VLM"))

import numpy as np                                       # noqa: E402
import mission as M                                      # noqa: E402
import mission_eval as ME                                # noqa: E402
from interface import parse_plan                         # noqa: E402
from stable_baselines3 import SAC                        # noqa: E402

CKPT = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
CELL = os.path.join(HERE, "out", "cells_v1_newprompt", "Qwen3.5-27B__both__id__rq1_both")
OUT = os.path.join(ROOT, "Experiment", "RQ1", "v1_newprompt", "gifs")
SEEDS = [0, 7, 13, 24, 42]

os.makedirs(OUT, exist_ok=True)
env = M.build_gcrl_env(); env.set_curriculum(1.0); env.max_steps = 1800
model = SAC.load(CKPT, env=env, device="auto")
rec = ME.Recorder(env.model, h=480, w=640)
print(f"GIF 저장 → {OUT}  (계획: both 셀, 컨트롤러: v1_startX_dist)")
for s in SEEDS:
    ob = M.gen_obstacles(s, "id")
    goals = parse_plan(os.path.join(CELL, f"seed{s}", "plan.json"))["goals"]
    rB = M.rollout_mission(env, model, s, ob, goals, sub_steps=200, rec=rec)
    tag = "safe" if rB["safe"] else ("succ" if rB["success"] else "fail")
    path = os.path.join(OUT, f"seed{s}_B_vlm_{tag}.gif")
    rec.save(path, stride=3)
    print(f"  seed{s}: subgoals={rB['n_sub']} pile도달={rB['stage1']} truck도달={rB['stage2']} "
          f"충돌={rB['collided']} → {os.path.basename(path)}")
rec.close(); env.close()
print("완료:", OUT)
