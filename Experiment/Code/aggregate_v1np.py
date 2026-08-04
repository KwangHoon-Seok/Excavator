"""새 프롬프트 + v1 GCRL RQ1 집계 → Experiment/RQ1/v1_newprompt/."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import aggregate_rqs as AG

TAG = "v1_newprompt"
AG.CELLS = os.path.join(HERE, "out", f"cells_{TAG}")
rq = {"RQ1": {
    "both": "Qwen3.5-27B__both__id__rq1_both",
    "rgb_only": "Qwen3.5-27B__rgb_only__id__rq1_rgb_only",
    "height_only": "Qwen3.5-27B__height_only__id__rq1_height_only",
}}
outdir = os.path.join(ROOT, "Experiment", "RQ1", TAG)
os.makedirs(outdir, exist_ok=True)
msg = AG.do_rq1(rq, outdir)
dp = os.path.join(outdir, "discussion.md")
if os.path.exists(dp):
    txt = open(dp).read().replace(
        "# RQ1 — 멀티모달 입력의 영향성",
        "# RQ1 — 멀티모달 입력의 영향성  (새 모달리티 프롬프트 + GCRL v1_startX_dist)")
    open(dp, "w").write(txt)
print(msg, "→", outdir)
