"""
지정 seed에 대해 A(단독 GCRL) / B(VLM+GCRL) 롤아웃을 GIF로 저장.
계획(plan.json)은 기존 셀에서 재사용(VLM 재호출 없음). GCRL 제어 env 시점 녹화:
  굴착기 + 장애물 박스 + 현재 목표(초록 구) + VLM 서브골 경로(파랑 작은 구).

  python Experiment/Code/gif_rollouts.py --seeds 0,1,3,7,2
"""
import os
import sys
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))   # Recorder
sys.path.insert(0, os.path.join(ROOT, "VLM"))                      # interface

import mission as M                                                # noqa: E402
import mission_eval as ME                                          # noqa: E402
from interface import parse_plan                                   # noqa: E402
from stable_baselines3 import SAC                                  # noqa: E402

CELL = os.path.join(HERE, "out", "cells", "Qwen3.5-35B-A3B__both__id__rq4_full")   # all-input(full) 셀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,3,7,2")
    ap.add_argument("--cell", default=CELL, help="plan.json 재사용할 셀 폴더")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "gifs"))
    ap.add_argument("--max_steps", type=int, default=1800)
    ap.add_argument("--sub_steps", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3, help="GIF 프레임 간격(클수록 작은 파일)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out, exist_ok=True)

    env = M.build_gcrl_env(); env.set_curriculum(1.0); env.max_steps = a.max_steps
    model = SAC.load(os.path.join(ROOT, "Model", "checkpoints", "Version4", "v4base.zip"),
                     env=env, device="auto")
    rec = ME.Recorder(env.model, h=480, w=640)
    print(f"GIF 저장 → {a.out}  (seeds={seeds}, plans={os.path.basename(a.cell)})")
    for s in seeds:
        ob = M.gen_obstacles(s, "id")
        goals = parse_plan(os.path.join(a.cell, f"seed{s}", "plan.json"))["goals"]
        rA = M.rollout_mission(env, model, s, ob, [], sub_steps=a.sub_steps, rec=rec)
        rec.save(os.path.join(a.out, f"seed{s}_A_baseline.gif"), stride=a.stride)
        rB = M.rollout_mission(env, model, s, ob, goals, sub_steps=a.sub_steps, rec=rec)
        rec.save(os.path.join(a.out, f"seed{s}_B_vlm.gif"), stride=a.stride)
        print(f"  seed {s}: A(pile={rA['stage1']} truck={rA['stage2']} coll={rA['collided']}) "
              f"B(pile={rB['stage1']} truck={rB['stage2']} coll={rB['collided']} sub={rB['n_sub']}) → GIF 2개")
    rec.close(); env.close()
    print("완료:", a.out)


if __name__ == "__main__":
    main()
