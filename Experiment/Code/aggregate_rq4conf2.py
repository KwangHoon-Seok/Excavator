import csv,numpy as np,os
from math import sqrt,erf
def stat(c):
    p=f"out/cells_rq4conf2/Qwen3.5-27B__both__id__{c}/results.csv"
    if not os.path.exists(p): return None
    rows=list(csv.DictReader(open(p))); n=len(rows); m=lambda k:np.mean([float(r[k]) for r in rows])
    return n,m('B_success'),m('B_safe'),m('B_coll'),m('B_stage1'),[int(float(r['B_success'])) for r in rows]
def ztest(a,b):
    n=len(a);k1,k2=sum(a),sum(b);p=(k1+k2)/(2*n);se=sqrt(p*(1-p)*(2/n)) if p not in(0,1) else 1e-9
    z=(k1-k2)/n/se; return 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
f=stat("full")
print(f"{'condition':18}{'success':10}{'safe':9}{'coll':8}{'pile':7}{'Δsuccess':10}{'p':8}")
print(f"{'full':18}{f[1]:.2f}      {f[2]:.2f}    {f[3]:.2f}   {f[4]:.2f}   --")
rows=[["full",f[1],f[2],f[3],f[4]]]
for c in ["drop2_kinematics","drop4_safety","drop8_examples"]:
    s=stat(c)
    if not s: print(f"{c:18}(진행 중)"); continue
    pv=ztest(f[5],s[5])
    print(f"{c:18}{s[1]:.2f}      {s[2]:.2f}    {s[3]:.2f}   {s[4]:.2f}   {s[1]-f[1]:+.2f}     {pv:.3f}{' *' if pv<0.05 else ''}")
    rows.append([c,s[1],s[2],s[3],s[4]])
os.makedirs("../RQ4",exist_ok=True)
with open("../RQ4/table_confusable_v2.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["condition","success","safe_success","collision","stage1"]); w.writerows([[r[0]]+[f"{x:.3f}" for x in r[1:]] for r in rows])
print("→ Experiment/RQ4/table_confusable_v2.csv")
