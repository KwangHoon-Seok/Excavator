import csv,numpy as np,os
def stat(p):
    if not os.path.exists(p): return None
    rows=list(csv.DictReader(open(p))); m=lambda k: np.mean([float(r[k]) for r in rows])
    return len(rows),m('B_success'),m('B_safe'),m('B_coll'),m('B_stage1')
full=stat("out/cells_rq1conf/Qwen3.5-27B__both__id__cart/results.csv")
print(f"{'condition':16}{'success':10}{'safe':9}{'coll':8}{'pile':7}{'Δsuccess':9}")
print(f"{'full':16}{full[1]:.2f}      {full[2]:.2f}    {full[3]:.2f}   {full[4]:.2f}   --")
rows=[["full"]+[full[1],full[2],full[3],full[4]]]
for c in ["no_kinematics","no_clearance","no_cot","no_examples","minimal"]:
    s=stat(f"out/cells_rq4conf/Qwen3.5-27B__both__id__{c}/results.csv")
    if not s: print(f"{c:16}(진행 중)"); continue
    print(f"{c:16}{s[1]:.2f}      {s[2]:.2f}    {s[3]:.2f}   {s[4]:.2f}   {s[1]-full[1]:+.2f}")
    rows.append([c,s[1],s[2],s[3],s[4]])
os.makedirs("../RQ4",exist_ok=True)
with open("../RQ4/table_confusable.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["condition","success","safe_success","collision","stage1"]); w.writerows([[r[0]]+[f"{x:.3f}" for x in r[1:]] for r in rows])
print("→ Experiment/RQ4/table_confusable.csv")
