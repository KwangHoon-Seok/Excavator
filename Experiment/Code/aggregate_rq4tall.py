"""RQ4-E2(tall) 집계 — 안전 지식 ablation이 binding되는 조건에서의 결과.

1차 DV = **계획 수준** 지표(GCRL/스윙 노이즈 배제):
  plan_safe   : HOME→서브골 전 구간 tip 경로가 모든 장애물 박스 밖(여유 0.05 m)
  plan_clear  : 그 경로의 최소 clearance 평균(m, 음수=관통)
  kin_valid   : 모든 서브골이 운동학 범위 안(r∈[1.04,4.46], z ≤ z_max(r))
2차 DV = 실행 지표(success/safe/collision/stage) — 하위 정책 요인 포함.

  python aggregate_rq4tall.py [--root out/cells_rq4tall] [--conds full,drop4_safety,safety_clean]
"""
import argparse, csv, json, os
from math import erf, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_TIP = (0.0, 3.62, 1.01)
ZTAB = [(1.0, 1.5, 2.12), (1.5, 2.0, 2.21), (2.0, 2.5, 2.29), (2.5, 3.0, 3.10),
        (3.0, 3.5, 3.19), (3.5, 4.0, 3.73), (4.0, 4.5, 4.00)]


def zmax(r):
    return next((z for a, b, z in ZTAB if a <= r < b), 4.00)


def _box_sd(p, c, h):
    """점 p와 축정렬 박스(center c, half h)의 부호거리(밖=+)."""
    d = np.abs(p - c) - h
    return float(np.linalg.norm(np.maximum(d, 0))) + min(float(np.max(d)), 0.0)


def plan_metrics(sg, obstacles, n=40):
    pts = [np.array(HOME_TIP, float)] + [np.array([s.get("x", 99), s.get("y", 99), s.get("z", 99)], float)
                                         for s in sg]
    clear = 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n):
            p = a + t * (b - a)
            for o in obstacles:
                clear = min(clear, _box_sd(p, np.array(o["center"], float), np.array(o["half"], float)))
    kin = all(1.04 <= np.hypot(p[0], p[1]) <= 4.46 and 0 <= p[2] <= zmax(np.hypot(p[0], p[1]))
              for p in pts[1:])
    return clear, int(clear > 0.05), int(kin), max(p[2] for p in pts[1:])


def cell(root, cond):
    d = os.path.join(HERE, root, f"Qwen3.5-27B__both__tall__{cond}")
    rows = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
    ex = {k: [float(r[k]) for r in rows] for k in
          ("B_success", "B_safe", "B_coll", "B_stage1", "B_stage2", "A_coll", "A_success")}
    cl, ps, kv, zt, nsg = [], [], [], [], []
    for r in rows:
        sd = os.path.join(d, f"seed{r['seed']}")
        try:
            pl = json.load(open(sd + "/plan.json")); sc = json.load(open(sd + "/scene.json"))
        except Exception:
            continue
        sg = pl.get("subgoals") or []
        if not sg:
            continue
        c, s, k, z = plan_metrics(sg, sc["obstacles"])
        cl.append(c); ps.append(s); kv.append(k); zt.append(z); nsg.append(len(sg))
    return dict(n=len(rows), prompt=rows[0]["prompt"], ex=ex,
                clear=np.mean(cl), plan_safe=ps, kin=kv, zmax=np.mean(zt), nsg=np.mean(nsg))


def ztest(a, b):
    n1, n2 = len(a), len(b); p = (sum(a) + sum(b)) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return 2 * (1 - 0.5 * (1 + erf(abs((np.mean(a) - np.mean(b)) / se) / sqrt(2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/cells_rq4tall")
    ap.add_argument("--conds", default="full,drop4_safety,safety_clean")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "RQ4", "table_tall.csv"))
    a = ap.parse_args()
    conds = a.conds.split(",")
    base = None
    hdr = ["condition", "prompt", "n", "plan_safe", "p_plan", "plan_clear_m", "kin_valid",
           "max_z", "n_subgoal", "success", "safe_success", "collision", "stage1", "stage2", "A_coll"]
    out = [hdr]
    print(f"{'condition':16}{'n':>4}{'plan_safe':>11}{'p':>8}{'clear':>8}{'kin_ok':>8}{'maxz':>7}"
          f"{'#sg':>6}{'succ':>7}{'safe':>7}{'coll':>7}{'s1':>6}{'s2':>6}")
    for c in conds:
        try:
            s = cell(a.root, c)
        except FileNotFoundError:
            print(f"{c:16}(진행 중)"); continue
        if base is None:
            base = s
        pv = ztest(s["plan_safe"], base["plan_safe"])
        e = s["ex"]
        print(f"{c:16}{s['n']:4d}{np.mean(s['plan_safe']):11.2f}{pv:8.3f}{s['clear']:+8.2f}"
              f"{np.mean(s['kin']):8.2f}{s['zmax']:7.2f}{s['nsg']:6.2f}"
              f"{np.mean(e['B_success']):7.2f}{np.mean(e['B_safe']):7.2f}{np.mean(e['B_coll']):7.2f}"
              f"{np.mean(e['B_stage1']):6.2f}{np.mean(e['B_stage2']):6.2f}")
        out.append([c, s["prompt"], s["n"], f"{np.mean(s['plan_safe']):.3f}", f"{pv:.3f}",
                    f"{s['clear']:.2f}", f"{np.mean(s['kin']):.3f}", f"{s['zmax']:.2f}", f"{s['nsg']:.2f}",
                    f"{np.mean(e['B_success']):.3f}", f"{np.mean(e['B_safe']):.3f}",
                    f"{np.mean(e['B_coll']):.3f}", f"{np.mean(e['B_stage1']):.3f}",
                    f"{np.mean(e['B_stage2']):.3f}", f"{np.mean(e['A_coll']):.3f}"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print("→", a.out)

    # rq4_subsection.tex의 tab:rq4-tall 자리표시자(\tbd)를 대체할 행을 그대로 출력
    label = {"full": "full", "drop4_safety": "$-$ safety, leaky (\\S4 only)",
             "safety_clean": "$-$ safety, leak-free"}
    tex = os.path.splitext(a.out)[0] + ".tex"
    lines = []
    for r in out[1:]:
        c, _, _, ps, pv, cl, kv, _, nsg, suc, _, coll = r[:12]
        p = "---" if c == conds[0] else pv
        lines.append(f"{label.get(c, c)} & {ps} & {p} & {cl} & {kv} & {nsg} & {suc} & {coll} \\\\")
    with open(tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("→", tex, "(tab:rq4-tall 본문 행)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
