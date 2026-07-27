"""
RQ1 재측정 — 새 모달리티 프롬프트(rq1_*, 섹션1~2만 모달별 교체) + 새 GCRL v1_startX_dist.
프롬프트가 바뀌어 VLM 계획을 새로 추론(옛 plan.json 재사용 안 함) → 새 경로에 저장.

VLM 계획 저장: out/cells_<tag>/<cellid>/seed<N>/plan.json  (mission_run이 seed별 저장)
롤아웃 결과   : out/cells_<tag>/<cellid>/results.csv

  # GPU0
  CUDA_VISIBLE_DEVICES=0 python rq1_remeasure.py --cells both,height_only
  # GPU1
  CUDA_VISIBLE_DEVICES=1 python rq1_remeasure.py --cells rgb_only
"""
import os
import sys
import json
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

PROMPTS = os.path.join(HERE, "prompts")
CKPT_V1 = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
MODEL = "Qwen/Qwen3.5-27B"
# 모달리티 → (프롬프트 파일, 셀 id). Cartesian 뷰: both/height는 Cartesian 프롬프트, rgb는 grid 없음(동일).
CELLS = {
    "both":        (os.path.join(PROMPTS, "rq1_cartesian.txt"),   "Qwen3.5-27B__both__id__cart"),
    "rgb_only":    (os.path.join(PROMPTS, "rq1_rgb_only.txt"),    "Qwen3.5-27B__rgb_only__id__cart"),
    "height_only": (os.path.join(PROMPTS, "rq1_cart_height.txt"), "Qwen3.5-27B__height_only__id__cart"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="both,rgb_only,height_only", help="쉼표구분 모달리티")
    ap.add_argument("--ckpt", default=CKPT_V1)
    ap.add_argument("--tag", default="v1_newprompt")
    ap.add_argument("--seeds_file", default=os.path.join(HERE, "seeds_mission.json"))
    ap.add_argument("--sub_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=1800)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--randomize", action="store_true", help="pile/truck 위치 seed별 랜덤")
    ap.add_argument("--confusable", action="store_true", help="흙더미·트럭을 박스로+장애물 크기 동일(높이맵 구별 불가)")
    a = ap.parse_args()

    import mission_run
    seeds = [int(s) for s in json.load(open(a.seeds_file))]
    out_root = os.path.join(HERE, "out", f"cells_{a.tag}")
    print(f"[remeasure] tag={a.tag} ckpt={os.path.basename(a.ckpt)} cells={a.cells} seeds={len(seeds)}", flush=True)
    for mod in a.cells.split(","):
        prompt, cid = CELLS[mod]
        out_dir = os.path.join(out_root, cid)
        if os.path.exists(os.path.join(out_dir, "results.csv")) and not a.refresh:
            print(f"[skip] {cid} (results.csv 있음)", flush=True)
            continue
        print(f"[run] {mod}  prompt={os.path.basename(prompt)} → {out_dir}", flush=True)
        mission_run.run(seeds, MODEL, prompt, mod, "id", a.ckpt, out_dir,
                        a.sub_steps, think=False, refresh=a.refresh, gif=False,
                        load_4bit=False, max_steps=a.max_steps, randomize=a.randomize,
                        confusable=a.confusable)
    print("[remeasure] done cells:", a.cells, flush=True)


if __name__ == "__main__":
    main()
