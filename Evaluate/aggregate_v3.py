"""
Version3 실험 집계 — 각 run의 fulltask.csv를 읽어 최근 성능을 표로 만들고
EXPERIMENTS.md의 '## 결과' 섹션을 자동 갱신. (TB 파싱 없이 CSV만 → 빠름)

fulltask.csv 컬럼(신형): step, fulltask_success, fulltask_mean_dist, safe_success, collision_rate
구형(3컬럼) run은 safe/collision을 '-'로 표기.

실행: conda run -n seok python Evaluate/aggregate_v3.py [--md Model/logs/Version3/EXPERIMENTS.md]
"""
import os, csv, glob, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_run(run_dir, last_n=5):
    """run의 fulltask.csv에서 최근 last_n 평가 평균 + 마지막 step."""
    fp = os.path.join(run_dir, "fulltask.csv")
    if not os.path.exists(fp):
        return None
    rows = list(csv.reader(open(fp)))
    if len(rows) < 2:
        return None
    hdr, data = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(hdr)}
    tail = data[-last_n:]

    def avg(name):
        if name not in col:
            return None
        vs = [float(r[col[name]]) for r in tail if len(r) > col[name] and r[col[name]] != ""]
        return sum(vs) / len(vs) if vs else None

    return dict(step=int(float(data[-1][0])),
                success=avg("fulltask_success"), dist=avg("fulltask_mean_dist"),
                safe=avg("safe_success"), coll=avg("collision_rate"))


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def fmt_step(s):
    return f"{s/1e6:.2f}M" if s >= 1e6 else f"{s//1000}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(ROOT, "Model", "logs", "Version3", "EXPERIMENTS.md"))
    ap.add_argument("--log_root", default=os.path.join(ROOT, "Model", "logs", "Version3"))
    a = ap.parse_args()

    runs = []
    for d in sorted(glob.glob(os.path.join(a.log_root, "*"))):
        if os.path.isdir(d):
            r = read_run(d)
            if r:
                r["name"] = os.path.basename(d)
                runs.append(r)
    # safe_success 내림차순 정렬(있는 것 우선)
    runs.sort(key=lambda r: (r["safe"] is not None, r["safe"] or -1), reverse=True)

    best = max((r["safe"] for r in runs if r["safe"] is not None), default=None)
    lines = ["| run | step | success | safe_success | collision | dist |",
             "|-----|------|---------|--------------|-----------|------|"]
    for r in runs:
        safe = fmt(r["safe"])
        if best is not None and r["safe"] == best:
            safe = f"**{safe}**"
        lines.append(f"| {r['name']} | {fmt_step(r['step'])} | {fmt(r['success'])} | "
                     f"{safe} | {fmt(r['coll'])} | {fmt(r['dist'])} |")
    table = "\n".join(lines)
    print(table)
    print(f"\nbest safe_success = {fmt(best)}" + (f" ({[r['name'] for r in runs if r['safe']==best][0]})" if best else ""))

    # md의 '## 결과' 섹션 테이블 교체
    if os.path.exists(a.md):
        txt = open(a.md).read().splitlines()
        out, i, replaced = [], 0, False
        while i < len(txt):
            out.append(txt[i])
            if txt[i].startswith("## 결과"):
                # 다음 '## ' 전까지 스킵하고, 헤더 설명 한 줄 뒤 표를 새로 삽입
                out.append("자동 집계(aggregate_v3.py). 최근 5평가 평균. best safe_success 굵게.\n")
                out.append(table + "\n")
                i += 1
                while i < len(txt) and not txt[i].startswith("## "):
                    i += 1
                replaced = True
                continue
            i += 1
        if replaced:
            open(a.md, "w").write("\n".join(out) + "\n")
            print(f"\n[갱신] {a.md} 의 '## 결과' 섹션")


if __name__ == "__main__":
    main()
