"""reach_ab.csv → 논문용 정제 figure(2패널).
(a) 거리별 success: A(단독 GCRL) vs B(VLM-hops+GCRL) — A가 붕괴하는 long-horizon 구간 음영.
(b) 거리별 VLM hop 수 — 먼 목표일수록 더 잘게 분해(적응적).
출력: Experiment/GCRL_compare/reach_ab_paper.{png,pdf}
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "Experiment", "GCRL_compare")
CSV = os.path.join(OUT, "reach_ab.csv")

rows = list(csv.DictReader(open(CSV)))
hd = np.array([float(r["home_dist"]) for r in rows])
A = np.array([int(r["A_success"]) for r in rows])
B = np.array([int(r["B_success"]) for r in rows])
H = np.array([int(r["n_hops"]) for r in rows])

BW = 0.8
bins = np.arange(0.4, hd.max() + BW, BW)
idx = np.digitize(hd, bins)
xs, ya, yb, hm, ns = [], [], [], [], []
for b in range(1, len(bins)):
    m = idx == b
    if m.sum() >= 3:
        xs.append((bins[b - 1] + bins[b]) / 2)
        ya.append(A[m].mean()); yb.append(B[m].mean())
        hm.append(H[m].mean()); ns.append(int(m.sum()))
xs = np.array(xs); ya = np.array(ya); yb = np.array(yb); hm = np.array(hm)

# A가 0.3 미만으로 붕괴하기 시작하는 지점(단독 GCRL long-horizon 한계) 추정
fail_x = None
for x, a in zip(xs, ya):
    if a < 0.3:
        fail_x = x - BW / 2
        break

CA, CB = "#8a8a8a", "#2f7ed8"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.5))

# 텍스트(제목/축라벨/범례/주석) 없이 저장 — 캡션·라벨은 LaTeX에서 추가.
# 데이터 스케일을 위한 축 눈금 숫자만 유지.

# ── (a) success vs distance ──
if fail_x is not None:
    ax1.axvspan(fail_x, xs.max() + BW / 2, color="#ffe1e1", alpha=0.55, zorder=0)
ax1.plot(xs, ya, "-o", color=CA, lw=2.4, ms=6, zorder=3)
ax1.plot(xs, yb, "-o", color=CB, lw=2.4, ms=6, zorder=3)
ax1.set_ylim(-0.03, 1.05); ax1.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
ax1.grid(alpha=0.3); ax1.set_axisbelow(True)
ax1.tick_params(labelsize=11)

# ── (b) hops vs distance ──
ax2.plot(xs, hm, "-D", color="#5a3fa0", lw=2.2, ms=6)
ax2.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
ax2.set_ylim(0, max(hm) + 1.2)
ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
ax2.tick_params(labelsize=11)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"reach_ab_paper.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved reach_ab_paper.png/pdf (no-text)  (A_fail_onset ~{fail_x}m)")
print("bins:")
for x, a, b, h, n in zip(xs, ya, yb, hm, ns):
    print(f"  ~{x:.1f}m: A={a:.2f} B={b:.2f} hops={h:.1f} (n={n})")
