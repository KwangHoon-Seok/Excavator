"""Claude(opus/sonnet)가 같은 gridmap을 읽어 물체 위치(특히 depth)를 Qwen27B보다 잘 짚는지 비교.
같은 프롬프트(rq1_both.txt)·같은 렌더 이미지 사용. claude CLI(기존 로그인) 경로=VLM_infer.
결과: out/opus_test/<model>/seed<N>/plan.json  +  pile/truck subgoal Δx,Δy 요약 출력.

  python opus_vs_qwen.py --model opus --seeds 0,7,13,24,42,55,66,88
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "VLM"))

import mission as M                      # noqa: E402
import VLM_infer as VI                   # noqa: E402
from interface import parse_plan         # noqa: E402

PILE = np.array(M.PILE_PICK); TRUCK = np.array(M.TRUCK_DUMP)
PROMPT = os.path.join(HERE, "prompts", "rq1_both.txt")   # Qwen both과 동일 프롬프트
SRC = os.path.join(HERE, "out", "cells_v1_newprompt", "Qwen3.5-27B__both__id__rq1_both")  # 렌더 재사용


def nearest(goals, GT):
    g = np.array(goals, float)
    return g[np.argmin(np.linalg.norm(g[:, :2] - GT[:2], axis=1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--seeds", default="0,7,13,24,42,55,66,88")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    outroot = os.path.join(HERE, "out", "opus_test", a.model)

    rows = []
    for s in seeds:
        seed_src = os.path.join(SRC, f"seed{s}")
        rgb = os.path.join(seed_src, "cabin_init.png")
        grid = os.path.join(seed_src, "gridmap.png")
        outd = os.path.join(outroot, f"seed{s}")
        os.makedirs(outd, exist_ok=True)
        # legend 파일도 seed_src에 있으므로 grid 옆에서 자동 로드됨
        try:
            VI.infer(M.INSTRUCTION, rgb, grid, model=a.model, timeout=a.timeout,
                     prompt=PROMPT, out=outd)
            goals = parse_plan(os.path.join(outd, "plan.json"))["goals"]
            if not goals:
                print(f"seed{s}: 계획 비어있음"); continue
            pg = nearest(goals, PILE); tg = nearest(goals, TRUCK)
            row = dict(seed=s, pdx=pg[0]-PILE[0], pdy=pg[1]-PILE[1],
                       tdx=tg[0]-TRUCK[0], tdy=tg[1]-TRUCK[1], nsub=len(goals))
            rows.append(row)
            print(f"seed{s}: pile Δx={row['pdx']:+.2f} Δy={row['pdy']:+.2f} | "
                  f"truck Δx={row['tdx']:+.2f} Δy={row['tdy']:+.2f} | nsub={row['nsub']}", flush=True)
        except Exception as e:
            print(f"seed{s}: 실패 {type(e).__name__}: {str(e)[:200]}", flush=True)

    if rows:
        import statistics as st
        def am(k): return st.mean(abs(r[k]) for r in rows)
        def sm(k): return st.mean(r[k] for r in rows)
        print(f"\n=== {a.model} 요약 (n={len(rows)}) ===")
        print(f"PILE : |Δx|={am('pdx'):.2f}  Δy(부호)={sm('pdy'):+.2f} |Δy|={am('pdy'):.2f}")
        print(f"TRUCK: |Δx|={am('tdx'):.2f}  Δy(부호)={sm('tdy'):+.2f} |Δy|={am('tdy'):.2f}")
        print(f"[대조] Qwen3.5-27B: PILE Δx+0.13 Δy+1.27 | TRUCK Δx-0.03 Δy+0.58")


if __name__ == "__main__":
    main()
