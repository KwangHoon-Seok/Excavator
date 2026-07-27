"""Qwen3.6-27B를 같은 7 seed로 테스트 — rq1_both 프롬프트, both 모달, v1 컨트롤러.
depth 판독 오차 + 실제 success(원본 계획 그대로, truck 보정 없음) 측정.
비교: Qwen3.5-27B, Opus.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "VLM"))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mission as M                       # noqa: E402
from interface import parse_plan          # noqa: E402
import mission_run                        # noqa: E402

PILE = np.array(M.PILE_PICK); TRUCK = np.array(M.TRUCK_DUMP)
SEEDS = [7, 13, 24, 42, 55, 66, 88]
MODEL = "Qwen/Qwen3.6-27B"
PROMPT = os.path.join(HERE, "prompts", "rq1_both.txt")
CKPT = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
OUT = os.path.join(HERE, "out", "cells_qwen36", "both")


def nearest(goals, GT):
    g = np.array(goals, float)
    return g[np.argmin(np.linalg.norm(g[:, :2] - GT[:2], axis=1))]


def main():
    rows = mission_run.run(SEEDS, MODEL, PROMPT, "both", "id", CKPT, OUT,
                           sub_steps=200, think=False, refresh=False, gif=False,
                           load_4bit=False, max_steps=1800)
    # depth 오차
    pe = []; te = []
    for s in SEEDS:
        pj = os.path.join(OUT, f"seed{s}", "plan.json")
        if not os.path.exists(pj):
            continue
        g = parse_plan(pj)["goals"]
        if not g:
            continue
        pg = nearest(g, PILE); tg = nearest(g, TRUCK)
        pe.append((pg[0]-PILE[0], pg[1]-PILE[1])); te.append((tg[0]-TRUCK[0], tg[1]-TRUCK[1]))
    pe = np.array(pe); te = np.array(te)
    r = lambda k: np.mean([row[k] for row in rows])
    print("\n================ Qwen3.6-27B (n=%d) ================" % len(rows))
    print("[위치 판독 오차]")
    print(f"  PILE : |Δx|={np.abs(pe[:,0]).mean():.2f}  Δy={pe[:,1].mean():+.2f} |Δy|={np.abs(pe[:,1]).mean():.2f}")
    print(f"  TRUCK: |Δx|={np.abs(te[:,0]).mean():.2f}  Δy={te[:,1].mean():+.2f} |Δy|={np.abs(te[:,1]).mean():.2f}")
    print("[실제 도달률 (원본 계획, v1 컨트롤러)]")
    print(f"  stage1={r('B_stage1'):.2f} stage2={r('B_stage2'):.2f} success={r('B_success'):.2f} "
          f"safe={r('B_safe'):.2f} coll={r('B_coll'):.2f}")
    print("\n[대조 — 같은 7 seed]")
    print("  Qwen3.5-27B: PILE Δy+1.27 | stage1=0.00 stage2=0.00 success=0.00")
    print("  Opus       : PILE Δy+0.01 | stage1=1.00 stage2=0.00 success=0.00 (truck 방위각 오차)")


if __name__ == "__main__":
    main()
