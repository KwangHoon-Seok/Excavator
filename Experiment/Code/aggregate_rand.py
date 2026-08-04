import os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); import aggregate_rqs as AG
AG.CELLS=os.path.join(HERE,"out","cells_rq1rand")
rq={"RQ1":{"both":"Qwen3.5-27B__both__id__cart","rgb_only":"Qwen3.5-27B__rgb_only__id__cart","height_only":"Qwen3.5-27B__height_only__id__cart"}}
outdir=os.path.join(ROOT,"Experiment","RQ1","randomized"); os.makedirs(outdir,exist_ok=True)
msg=AG.do_rq1(rq,outdir)
dp=os.path.join(outdir,"discussion.md")
if os.path.exists(dp): open(dp,"w").write(open(dp).read().replace("# RQ1 — 멀티모달 입력의 영향성","# RQ1 — 멀티모달 (pile/truck 위치 랜덤화, Cartesian, v1)"))
print(msg,"→",outdir)
