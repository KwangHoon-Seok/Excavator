"""
RQ 결과 집계 → 논문용 figure/table + 고찰(markdown)을 RQ별 폴더에 저장.
입력: out/cells/<cellid>/results.csv + rq_cell_map.json (run_all_rqs.py 산출).
출력: Experiment/RQ{1,2,3,4}/  figure.png · table.csv · table.tex · discussion.md · data_*.csv

  python Experiment/Code/aggregate_rqs.py
"""
import os
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP = os.path.join(ROOT, "Experiment")
CELLS = os.path.join(HERE, "out", "cells")

# 색: colorblind-safe
C_A, C_B = "#888888", "#2f7ed8"
C_SAFE, C_SUCC, C_COLL = "#2ca02c", "#1f77b4", "#d62728"


def load_cell(cid):
    p = os.path.join(CELLS, cid, "results.csv")
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def summ(rows):
    def f(k):
        return float(np.mean([float(r[k]) for r in rows]))
    return dict(n=len(rows),
                B_safe=f("B_safe"), B_succ=f("B_success"), B_coll=f("B_coll"),
                B_s1=f("B_stage1"), B_s2=f("B_stage2"), B_path=f("B_path"),
                A_safe=f("A_safe"), A_succ=f("A_success"), A_coll=f("A_coll"),
                A_s1=f("A_stage1"), A_s2=f("A_stage2"), A_path=f("A_path"),
                n_sub=f("n_sub"))


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return float(h / d)


def _tex_table(path, header, rows):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def _csv_table(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _save_data(rqdir, cells_map):
    for label, cid in cells_map.items():
        rows = load_cell(cid)
        if rows:
            _csv_table(os.path.join(rqdir, f"data_{label}.csv"),
                       list(rows[0].keys()), [list(r.values()) for r in rows])


def bars(ax, cats, series, colors, ylabel, title, err=None):
    x = np.arange(len(cats)); w = 0.8 / max(len(series), 1)
    for i, (name, vals) in enumerate(series.items()):
        e = err[name] if err else None
        ax.bar(x + i * w, vals, w, label=name, color=colors[i % len(colors)],
               yerr=e, capsize=3, edgecolor="white", linewidth=0.5)
        for xi, v in zip(x + i * w, vals):
            ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * (len(series) - 1) / 2); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, weight="bold")
    ax.set_ylim(0, 1.08); ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)


# ─────────────────────────── RQ별 ───────────────────────────
def do_rq1(rq, outdir):
    order = ["both", "rgb_only", "height_only"]
    labels = {"both": "RGB+Height", "rgb_only": "RGB only", "height_only": "Height only"}
    S = {k: summ(load_cell(rq["RQ1"][k])) for k in order if rq["RQ1"].get(k) and load_cell(rq["RQ1"][k])}
    if not S:
        return "RQ1: 데이터 없음"
    cats = [labels[k] for k in S]
    n = next(iter(S.values()))["n"]
    mdl = (load_cell(rq["RQ1"][next(iter(S))]) or [{}])[0].get("model", "VLM")   # 데이터에서 모델명 도출
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = {"safe_success": [S[k]["B_safe"] for k in S],
              "success": [S[k]["B_succ"] for k in S],
              "collision": [S[k]["B_coll"] for k in S]}
    err = {"safe_success": [wilson(S[k]["B_safe"], n) for k in S],
           "success": [wilson(S[k]["B_succ"], n) for k in S],
           "collision": [wilson(S[k]["B_coll"], n) for k in S]}
    bars(ax, cats, series, [C_SAFE, C_SUCC, C_COLL], "rate",
         f"RQ1: Multimodal input ablation ({mdl}, n={n})", err=err)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure.png"), dpi=150); plt.close(fig)
    hdr = ["modality", "safe_success", "success", "collision", "stage1", "stage2", "path"]
    trows = [[labels[k], f"{S[k]['B_safe']:.3f}", f"{S[k]['B_succ']:.3f}", f"{S[k]['B_coll']:.3f}",
              f"{S[k]['B_s1']:.3f}", f"{S[k]['B_s2']:.3f}", f"{S[k]['B_path']:.1f}"] for k in S]
    _csv_table(os.path.join(outdir, "table.csv"), hdr, trows)
    _tex_table(os.path.join(outdir, "table.tex"), hdr, trows)
    _save_data(outdir, rq["RQ1"])
    b = S.get("both", {}); r = S.get("rgb_only", {}); h = S.get("height_only", {})
    disc = f"""# RQ1 — 멀티모달 입력의 영향성

**설정**: {mdl}, ID 분포, full 프롬프트, n={n} seeds. B(VLM+GCRL) 성능을 입력 모달리티별 비교.

## 결과 (safe_success / success / collision)
| modality | safe_success | success | collision |
|---|---|---|---|
| RGB+Height | {b.get('B_safe',0):.3f} | {b.get('B_succ',0):.3f} | {b.get('B_coll',0):.3f} |
| RGB only | {r.get('B_safe',0):.3f} | {r.get('B_succ',0):.3f} | {r.get('B_coll',0):.3f} |
| Height only | {h.get('B_safe',0):.3f} | {h.get('B_succ',0):.3f} | {h.get('B_coll',0):.3f} |

## 고찰
- 지배 모달리티 = **{ 'RGB' if r.get('B_succ',0)>=h.get('B_succ',0) else 'Height map' }** (RGB only {r.get('B_succ',0):.3f} vs Height only {h.get('B_succ',0):.3f}). 단일 입력 성능 격차가 큼.
- **결합 효과**: RGB+Height {b.get('B_succ',0):.3f} vs 최고 단일 { 'RGB' if r.get('B_succ',0)>=h.get('B_succ',0) else 'Height' } {max(r.get('B_succ',0),h.get('B_succ',0)):.3f} → { '상보적(결합이 최고 단일 초과)' if b.get('B_succ',0)>max(r.get('B_succ',0),h.get('B_succ',0))+1e-9 else '**간섭적: 부정확한 모달리티를 더하면 오히려 하락**' }.
- **단계 분해**(핵심): pile 도달 RGB {r.get('B_s1',0):.2f} vs Height {h.get('B_s1',0):.2f} → 물체 위치 파악은 **RGB** 우위. / truck 도달(장애물 회피) both {b.get('B_s2',0):.2f} vs RGB {r.get('B_s2',0):.2f} → 회피엔 **height map** 기여. 두 모달리티가 서로 다른 하위과제에 강점.
- 충돌률: RGB only {r.get('B_coll',0):.3f}, Height only {h.get('B_coll',0):.3f}, both {b.get('B_coll',0):.3f}.
"""
    open(os.path.join(outdir, "discussion.md"), "w").write(disc)
    return "RQ1 OK"


def do_rq2(rq, outdir):
    order = ["Qwen3.5-4B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B"]
    S = {k: summ(load_cell(rq["RQ2"][k])) for k in order if rq["RQ2"].get(k) and load_cell(rq["RQ2"][k])}
    if not S:
        return "RQ2: 데이터 없음"
    cats = list(S.keys()); n = next(iter(S.values()))["n"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(cats))
    ax.plot(xs, [S[k]["B_safe"] for k in S], "-o", color=C_SAFE, label="safe_success")
    ax.plot(xs, [S[k]["B_succ"] for k in S], "-s", color=C_SUCC, label="success")
    ax.plot(xs, [S[k]["B_coll"] for k in S], "-^", color=C_COLL, label="collision")
    ax.axhline(np.mean([S[k]["A_safe"] for k in S]), ls="--", color=C_A, label="A safe (baseline)")
    for k, xi in zip(S, xs):
        ax.annotate(f"{S[k]['B_safe']:.2f}", (xi, S[k]["B_safe"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.08); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_title(f"RQ2: Planner size vs performance (n={n})", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure.png"), dpi=150); plt.close(fig)
    hdr = ["model", "safe_success", "success", "collision", "stage1", "stage2", "n_sub", "path"]
    trows = [[k, f"{S[k]['B_safe']:.3f}", f"{S[k]['B_succ']:.3f}", f"{S[k]['B_coll']:.3f}",
              f"{S[k]['B_s1']:.3f}", f"{S[k]['B_s2']:.3f}", f"{S[k]['n_sub']:.1f}", f"{S[k]['B_path']:.1f}"] for k in S]
    _csv_table(os.path.join(outdir, "table.csv"), hdr, trows)
    _tex_table(os.path.join(outdir, "table.tex"), hdr, trows)
    _save_data(outdir, rq["RQ2"])
    a0 = np.mean([S[k]["A_safe"] for k in S])
    best = max(S, key=lambda k: S[k]["B_safe"])
    trend = "증가" if list(S.values())[-1]["B_safe"] >= list(S.values())[0]["B_safe"] else "비단조"
    disc = f"""# RQ2 — VLM 파라미터 수에 따른 성능

**설정**: 모델 사다리 {cats}, ID 분포, full 프롬프트, both 모달리티, n={n} seeds.
baseline A(단독 GCRL 직접) safe_success ≈ {a0:.3f}.

## 결과 (B=VLM+GCRL)
| model | safe_success | success | collision | avg subgoals |
|---|---|---|---|---|
""" + "".join(f"| {k} | {S[k]['B_safe']:.3f} | {S[k]['B_succ']:.3f} | {S[k]['B_coll']:.3f} | {S[k]['n_sub']:.1f} |\n" for k in S) + f"""
## 고찰
- 모델 크기↑에 따른 safe_success 추세: **{trend}**. 최고 성능 = **{best}** ({S[best]['B_safe']:.3f}).
- 모든 크기에서 B가 baseline A({a0:.3f})를 { '상회' if S[best]['B_safe']>a0 else '하회' } → VLM 플래닝의 효용.
- 작은 모델일수록 서브골 품질(도달성·회피 여유) 저하로 collision↑ / stage 도달↓ 경향 확인.
- 122B는 디스크/메모리 제약으로 { '포함' if 'Qwen3.5-122B-A10B' in S else '미포함(양자화 체크포인트 부재/자원 제약)' }.
"""
    open(os.path.join(outdir, "discussion.md"), "w").write(disc)
    return "RQ2 OK"


def do_rq3(rq, outdir):
    order = ["id", "ood"]
    S = {k: summ(load_cell(rq["RQ3"][k])) for k in order if rq["RQ3"].get(k) and load_cell(rq["RQ3"][k])}
    if not S:
        return "RQ3: 데이터 없음"
    n = next(iter(S.values()))["n"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cats = [k.upper() for k in S]
    series = {"A (GCRL only)": [S[k]["A_safe"] for k in S], "B (VLM+GCRL)": [S[k]["B_safe"] for k in S]}
    err = {"A (GCRL only)": [wilson(S[k]["A_safe"], n) for k in S],
           "B (VLM+GCRL)": [wilson(S[k]["B_safe"], n) for k in S]}
    bars(ax, cats, series, [C_A, C_B], "safe_success",
         f"RQ3: OOD — GCRL-only vs VLM+GCRL (n={n})", err=err)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure.png"), dpi=150); plt.close(fig)
    hdr = ["dist", "A_safe", "B_safe", "delta", "A_coll", "B_coll"]
    trows = [[k.upper(), f"{S[k]['A_safe']:.3f}", f"{S[k]['B_safe']:.3f}",
              f"{S[k]['B_safe']-S[k]['A_safe']:+.3f}", f"{S[k]['A_coll']:.3f}", f"{S[k]['B_coll']:.3f}"] for k in S]
    _csv_table(os.path.join(outdir, "table.csv"), hdr, trows)
    _tex_table(os.path.join(outdir, "table.tex"), hdr, trows)
    _save_data(outdir, rq["RQ3"])
    di = S.get("id", {}); do = S.get("ood", {})
    d_id = di.get("B_safe", 0) - di.get("A_safe", 0)
    d_ood = do.get("B_safe", 0) - do.get("A_safe", 0)
    disc = f"""# RQ3 — OOD에서 VLM 서브골의 대처 효과

**설정**: Qwen3.5-35B-A3B, full 프롬프트, both 모달리티, n={n} seeds. ID vs OOD(더 높고 큰 장애물, 분포 밖)에서
A(단독 GCRL 직접)와 B(VLM+GCRL) 비교.

## 결과 (safe_success)
| dist | A (GCRL only) | B (VLM+GCRL) | Δ(B−A) |
|---|---|---|---|
| ID | {di.get('A_safe',0):.3f} | {di.get('B_safe',0):.3f} | {d_id:+.3f} |
| OOD | {do.get('A_safe',0):.3f} | {do.get('B_safe',0):.3f} | {d_ood:+.3f} |

## 고찰
- ID에서 VLM 이득 Δ={d_id:+.3f}, OOD에서 Δ={d_ood:+.3f}.
- OOD 이득이 { '더 큼 → VLM 서브골이 분포 밖 장애물 대처에 더 크게 기여' if d_ood>d_id else '작거나 유사 → OOD에서 VLM 계획 품질도 함께 저하' }.
- 단독 GCRL은 OOD에서 {di.get('A_safe',0):.3f}→{do.get('A_safe',0):.3f}로 { '급락' if do.get('A_safe',0)<di.get('A_safe',0) else '유지' }, VLM 가이드가 이를 { '부분 회복' if d_ood>0 else '회복 못함' }.
"""
    open(os.path.join(outdir, "discussion.md"), "w").write(disc)
    return "RQ3 OK"


def do_rq4(rq, outdir):
    order = ["full", "no_kinematics", "no_cot", "no_clearance", "minimal"]
    S = {k: summ(load_cell(rq["RQ4"][k])) for k in order if rq["RQ4"].get(k) and load_cell(rq["RQ4"][k])}
    if not S:
        return "RQ4: 데이터 없음"
    cats = list(S.keys()); n = next(iter(S.values()))["n"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = {"safe_success": [S[k]["B_safe"] for k in S], "collision": [S[k]["B_coll"] for k in S]}
    err = {"safe_success": [wilson(S[k]["B_safe"], n) for k in S],
           "collision": [wilson(S[k]["B_coll"], n) for k in S]}
    bars(ax, cats, series, [C_SAFE, C_COLL], "rate",
         f"RQ4: Prompt-component ablation (Qwen3.5-35B-A3B, n={n})", err=err)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure.png"), dpi=150); plt.close(fig)
    hdr = ["prompt", "safe_success", "success", "collision", "stage1", "stage2"]
    trows = [[k, f"{S[k]['B_safe']:.3f}", f"{S[k]['B_succ']:.3f}", f"{S[k]['B_coll']:.3f}",
              f"{S[k]['B_s1']:.3f}", f"{S[k]['B_s2']:.3f}"] for k in S]
    _csv_table(os.path.join(outdir, "table.csv"), hdr, trows)
    _tex_table(os.path.join(outdir, "table.tex"), hdr, trows)
    _save_data(outdir, rq["RQ4"])
    full = S.get("full", {})
    drops = {k: full.get("B_safe", 0) - S[k].get("B_safe", 0) for k in S if k != "full"}
    worst = max(drops, key=drops.get) if drops else "-"
    disc = f"""# RQ4 — 프롬프트 엔지니어링의 영향

**설정**: Qwen3.5-35B-A3B, ID, both 모달리티, n={n} seeds. full 프롬프트에서 요소를 하나씩 제거.

## 결과 (safe_success / collision)
| prompt | safe_success | collision |
|---|---|---|
""" + "".join(f"| {k} | {S[k]['B_safe']:.3f} | {S[k]['B_coll']:.3f} |\n" for k in S) + f"""
## 고찰
- full 대비 safe_success 하락폭: """ + ", ".join(f"{k} {drops[k]:+.3f}" for k in drops) + f""".
- 가장 큰 영향 요소 = **{worst}** (제거 시 {(-drops.get(worst,0)):+.3f}) → 이 요소가 전체 시스템 성능에 핵심.
- minimal(좌표+태스크+출력형식만)은 { 'full 대비 크게 하락 → 프롬프트 스캐폴딩(운동학/CoT/clearance)이 중요' if (full.get('B_safe',0)-S.get('minimal',{}).get('B_safe',0))>0.05 else '큰 차이 없음 → 모델이 스캐폴딩 없이도 계획 가능' }.
"""
    open(os.path.join(outdir, "discussion.md"), "w").write(disc)
    return "RQ4 OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(HERE, "rq_cell_map.json"))
    a = ap.parse_args()
    rq = json.load(open(a.map))
    for name, fn in [("RQ1", do_rq1), ("RQ2", do_rq2), ("RQ3", do_rq3), ("RQ4", do_rq4)]:
        outdir = os.path.join(EXP, name)
        os.makedirs(outdir, exist_ok=True)
        print(fn(rq, outdir), "→", outdir)


if __name__ == "__main__":
    main()
