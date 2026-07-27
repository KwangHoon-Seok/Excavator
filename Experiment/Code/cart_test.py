"""Cartesian top-down 뷰 + Cartesian 프롬프트로 Qwen3.5-27B 재측정 (같은 7 seed).
인지 개선 검증: 장애물 recall + pile/truck depth 오차 → 극좌표 대비.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "VLM"))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mission as M
from interface import parse_plan
import qwen_planner
from render_cartesian import render_cartesian

PILE = np.array(M.PILE_PICK); TRUCK = np.array(M.TRUCK_DUMP)
SEEDS = [7, 13, 24, 42, 55, 66, 88]
MODEL = "Qwen/Qwen3.5-27B"
PROMPT = os.path.join(HERE, "prompts", "rq1_cartesian.txt")
SRC = os.path.join(HERE, "out", "cells_v1_newprompt", "Qwen3.5-27B__both__id__rq1_both")
OUT = os.path.join(HERE, "out", "cells_cart27", "both")
MTOL = 0.6


def nearest(goals, GT):
    g = np.array(goals, float); return g[np.argmin(np.linalg.norm(g[:, :2] - GT[:2], axis=1))]


def main():
    for s in SEEDS:
        sd = os.path.join(OUT, f"seed{s}"); os.makedirs(sd, exist_ok=True)
        rgb = os.path.join(SRC, f"seed{s}", "cabin_init.png")
        npz = os.path.join(SRC, f"seed{s}", "gridmap.npz")
        cart = os.path.join(sd, "gridmap_cart.png")
        render_cartesian(npz, cart)
        try:
            qwen_planner.infer(M.INSTRUCTION, rgb, cart, model=MODEL, prompt=PROMPT,
                               out=sd, modality="both", think=False)
            print(f"seed{s}: planned", flush=True)
        except Exception as e:
            print(f"seed{s}: 실패 {type(e).__name__} {str(e)[:150]}", flush=True)

    # 측정: 장애물 recall + depth
    gt_tot = det = fp = 0; oerr = []; pe = []; te = []
    for s in SEEDS:
        pj = os.path.join(OUT, f"seed{s}", "plan.json")
        if not os.path.exists(pj): continue
        pl = json.load(open(pj)); g = parse_plan(pj)["goals"]
        gt = M.gen_obstacles(s, "id"); gt_xy = [np.array(o["center"][:2]) for o in gt]; gt_tot += len(gt)
        preds = [o for o in pl.get("objects", []) if o.get("type") == "obstacle"]; used = set()
        for o in preds:
            try: p = np.array([float(o["x"]), float(o["y"])])
            except: fp += 1; continue
            ds = [(np.linalg.norm(p - gv), i) for i, gv in enumerate(gt_xy) if i not in used]
            if ds and min(ds)[0] <= MTOL: d, i = min(ds); used.add(i); det += 1; oerr.append(d)
            else: fp += 1
        if g:
            pg = nearest(g, PILE); tg = nearest(g, TRUCK)
            pe.append((pg[0]-PILE[0], pg[1]-PILE[1])); te.append((tg[0]-TRUCK[0], tg[1]-TRUCK[1]))
    pe = np.array(pe); te = np.array(te)
    print("\n============ Cartesian + Qwen3.5-27B (n=%d) ============" % len(pe))
    print("[장애물 인지]")
    print(f"  recall={det/gt_tot:.2f} (탐지 {det}/{gt_tot})  위치오차={np.mean(oerr) if oerr else float('nan'):.2f}m  허위={fp}")
    print("[타깃 위치 오차]")
    print(f"  PILE : |Δx|={np.abs(pe[:,0]).mean():.2f}  Δy={pe[:,1].mean():+.2f} |Δy|={np.abs(pe[:,1]).mean():.2f}")
    print(f"  TRUCK: |Δx|={np.abs(te[:,0]).mean():.2f}  Δy={te[:,1].mean():+.2f} |Δy|={np.abs(te[:,1]).mean():.2f}")
    print("\n[대조 — 극좌표 Qwen3.5-27B, 같은 7 seed]")
    print("  장애물 recall=0.24 위치오차0.32 허위6 | PILE Δy+1.27 | TRUCK Δy+0.58")


if __name__ == "__main__":
    main()
