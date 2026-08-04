"""RQ4 전체(3배치) 통합 집계 → Experiment/RQ4/table_all.csv
  batch: id_normal(랜덤 ID) / confusable_v1(rq1_cartesian 계열) / confusable_v2(rq4_full 재정리 계열)
  각 조건 vs 해당 배치 full의 2-proportion z-test p값 포함, 계획 품질 진단(plan→pile 최소거리) 포함.
"""
import csv, json, os
import numpy as np
from math import sqrt, erf

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = {
    "id_normal": [("full", "out/cells_rq1rand/Qwen3.5-27B__both__id__cart"),
                  *[(c, f"out/cells_rq4/Qwen3.5-27B__both__id__{c}") for c in
                    ["no_kinematics", "no_clearance", "no_cot", "no_examples", "minimal"]]],
    "confusable_v1": [("full", "out/cells_rq1conf/Qwen3.5-27B__both__id__cart"),
                      *[(c, f"out/cells_rq4conf/Qwen3.5-27B__both__id__{c}") for c in
                        ["no_kinematics", "no_clearance", "no_cot", "no_examples", "minimal"]]],
    "confusable_v2": [(c, f"out/cells_rq4conf2/Qwen3.5-27B__both__id__{c}") for c in
                      ["full", "drop2_kinematics", "drop4_safety", "drop8_examples"]],
}


def cell(d):
    rows = list(csv.DictReader(open(os.path.join(HERE, d, "results.csv"))))
    v = lambda k: [float(r[k]) for r in rows]
    mind, nobs = [], []
    for r in rows:
        sd = os.path.join(HERE, d, f"seed{r['seed']}")
        try:
            pl = json.load(open(sd + "/plan.json")); sc = json.load(open(sd + "/scene.json"))
        except Exception:
            continue
        sg = pl.get("subgoals") or []
        if not sg:
            continue
        px, py, _ = sc["pile_pick"]
        mind.append(min(np.hypot(s.get("x", 99) - px, s.get("y", 99) - py) for s in sg))
        nobs.append(sum(1 for o in (pl.get("objects") or []) if o.get("type") == "obstacle"))
    return dict(n=len(rows), prompt=rows[0]["prompt"], succ=v("B_success"), safe=v("B_safe"),
                coll=v("B_coll"), s1=v("B_stage1"), s2=v("B_stage2"),
                mind=np.mean(mind), hit=np.mean([x <= 0.5 for x in mind]), nobs=np.mean(nobs))


def ztest(a, b):
    n1, n2 = len(a), len(b); p = (sum(a) + sum(b)) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0: return 1.0
    z = (np.mean(a) - np.mean(b)) / se
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


out = [["batch", "condition", "prompt", "n", "success", "safe_success", "collision",
        "stage1_pile", "stage2_truck", "d_success", "p_success", "plan_pile_dist_m", "plan_pile_hit", "n_obstacle_listed"]]
for b, conds in BATCH.items():
    base = None
    print(f"\n##### {b}")
    print(f"{'cond':18}{'n':>4}{'succ':>7}{'safe':>7}{'coll':>7}{'pile':>7}{'truck':>7}{'Δsucc':>8}{'p':>7}{'plan→pile':>11}{'hit':>6}{'obs':>6}")
    for c, d in conds:
        s = cell(d)
        if base is None: base = s
        dd, pv = np.mean(s["succ"]) - np.mean(base["succ"]), ztest(s["succ"], base["succ"])
        print(f"{c:18}{s['n']:4d}{np.mean(s['succ']):7.2f}{np.mean(s['safe']):7.2f}{np.mean(s['coll']):7.2f}"
              f"{np.mean(s['s1']):7.2f}{np.mean(s['s2']):7.2f}{dd:+8.2f}{pv:7.3f}{s['mind']:11.2f}{s['hit']:6.2f}{s['nobs']:6.2f}")
        out.append([b, c, s["prompt"], s["n"], f"{np.mean(s['succ']):.3f}", f"{np.mean(s['safe']):.3f}",
                    f"{np.mean(s['coll']):.3f}", f"{np.mean(s['s1']):.3f}", f"{np.mean(s['s2']):.3f}",
                    f"{dd:+.3f}", f"{pv:.3f}", f"{s['mind']:.2f}", f"{s['hit']:.2f}", f"{s['nobs']:.2f}"])
p = os.path.join(HERE, "..", "RQ4", "table_all.csv")
with open(p, "w", newline="") as f:
    csv.writer(f).writerows(out)
print("\n→ Experiment/RQ4/table_all.csv")
