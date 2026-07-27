import os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); import aggregate_rqs as AG
AG.CELLS=os.path.join(HERE,"out","cells_rq1conf")
rq={"RQ1":{"both":"Qwen3.5-27B__both__id__cart","rgb_only":"Qwen3.5-27B__rgb_only__id__cart","height_only":"Qwen3.5-27B__height_only__id__cart"}}
outdir=os.path.join(ROOT,"Experiment","RQ1","confusable"); os.makedirs(outdir,exist_ok=True)
print(AG.do_rq1(rq,outdir),"→",outdir)
