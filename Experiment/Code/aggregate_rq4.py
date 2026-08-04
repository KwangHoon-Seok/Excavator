import csv,numpy as np,os,sys
def stat(path):
    if not os.path.exists(path): return None
    rows=list(csv.DictReader(open(path))); m=lambda k: np.mean([float(r[k]) for r in rows])
    return len(rows),m('B_success'),m('B_safe'),m('B_coll'),m('B_stage1'),m('B_stage2')
conds=[("full","out/cells_rq1rand/Qwen3.5-27B__both__id__cart/results.csv"),
       ("no_kinematics","out/cells_rq4/Qwen3.5-27B__both__id__no_kinematics/results.csv"),
       ("no_clearance","out/cells_rq4/Qwen3.5-27B__both__id__no_clearance/results.csv"),
       ("no_cot","out/cells_rq4/Qwen3.5-27B__both__id__no_cot/results.csv"),
       ("no_examples","out/cells_rq4/Qwen3.5-27B__both__id__no_examples/results.csv"),
       ("minimal","out/cells_rq4/Qwen3.5-27B__both__id__minimal/results.csv")]
full=stat(conds[0][1])
print(f"{'condition':16}{'success':10}{'safe':9}{'coll':8}{'pile':7}{'truck':7}{'Δsuccess':9}")
rows=[]
for name,p in conds:
    s=stat(p)
    if not s: print(f"{name:16}(진행 중)"); continue
    n,suc,safe,coll,s1,s2=s
    d=suc-full[1] if full else 0
    print(f"{name:16}{suc:.2f}      {safe:.2f}    {coll:.2f}   {s1:.2f}  {s2:.2f}  {d:+.2f}")
    rows.append([name,suc,safe,coll,s1,s2])
os.makedirs("../RQ4",exist_ok=True)
with open("../RQ4/table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["condition","success","safe_success","collision","stage1","stage2"]); w.writerows([[r[0]]+[f"{x:.3f}" for x in r[1:]] for r in rows])
print("→ Experiment/RQ4/table.csv")
