"""극좌표 height grid(npz: H,X,Y) → Cartesian top-down (x,y) 높이맵 PNG.
VLM이 (x,y)를 직접 읽게 하여 θ,r 삼각변환·radial 오독 제거.
  x축 = 좌우 (− left | + right, m),  y축 = 전방거리 (m),  굴착기 = 원점.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BANDS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
COLORS = ["#d9d9d9", "#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c"]


def render_cart_from_grid(grid, out_png, res=0.05, px=(900, 760)):
    """polar_height_grid가 반환한 grid dict(H,r_max,half_span_deg)에서 직접 Cartesian 렌더."""
    return _render(np.asarray(grid["H"]), float(grid["r_max"]),
                   float(grid["half_span_deg"]), out_png, res, px)


def render_cartesian(npz, out_png, res=0.05, px=(900, 760)):
    import json
    d = np.load(npz)
    H = d["H"]                      # (n_theta, n_r)
    meta_p = npz.replace(".npz", ".json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p)); r_max = float(m["r_max"]); half = float(m["half_span_deg"])
    else:                           # npz만 있으면 셀 중심에서 유도
        r_max = float(np.max(d["r_c"])) + 0.05
        half = float(np.degrees(np.max(np.abs(d["theta_c"]))))
    return _render(H, r_max, half, out_png, res, px)


def _render(H, r_max, half, out_png, res=0.05, px=(900, 760)):
    n_th, n_r = H.shape
    half_rad = np.deg2rad(half)
    # 규칙적 Cartesian 격자 → 각 픽셀을 극좌표로 변환해 최근접 polar 셀 H 조회
    xs = np.arange(-r_max, r_max + res, res)
    ys = np.arange(0.0, r_max + res, res)
    GX, GY = np.meshgrid(xs, ys)
    R = np.hypot(GX, GY)
    TH = np.arctan2(GX, GY)         # +x=오른쪽 → θ>0 오른쪽 (극좌표와 동일 규약)
    C = np.full(GX.shape, np.nan)
    inside = (R <= r_max) & (np.abs(TH) <= half_rad)
    ri = np.clip(np.round(R / r_max * (n_r - 1)).astype(int), 0, n_r - 1)
    ti = np.clip(np.round((TH + half_rad) / (2 * half_rad) * (n_th - 1)).astype(int), 0, n_th - 1)
    C[inside] = H[ti[inside], ri[inside]]

    cmap = ListedColormap(COLORS); cmap.set_bad("#ffffff")
    norm = BoundaryNorm(BANDS, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(px[0] / 100, px[1] / 100), dpi=100)
    ax.pcolormesh(xs, ys, C, cmap=cmap, norm=norm, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlim(-r_max, r_max); ax.set_ylim(0, r_max)
    ax.set_xticks(np.arange(-4, 4.01, 1.0)); ax.set_yticks(np.arange(0, r_max + 0.01, 1.0))
    ax.grid(True, color="0.55", lw=0.8, alpha=0.9)
    ax.set_xlabel("x  (m)   [−  left  |  +  right]", fontsize=11)
    ax.set_ylabel("y  forward distance (m)", fontsize=11)
    ax.plot(0, 0, marker="^", ms=13, color="black", zorder=5)      # 굴착기(원점)
    ax.annotate("excavator (0,0)", (0, 0), (0.1, 0.12), fontsize=9)
    labels = [f"{BANDS[i]:g}~{BANDS[i+1]:g}" for i in range(len(COLORS))]
    labels[0] += " ground"
    handles = [Patch(facecolor=COLORS[i], edgecolor="0.3", label=labels[i]) for i in range(len(COLORS))]
    ax.legend(handles=handles, title="height h (m)", loc="lower right", fontsize=7, title_fontsize=8,
              ncol=2, framealpha=0.95)
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
    return out_png


if __name__ == "__main__":
    npz = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "cart.png"
    print("saved", render_cartesian(npz, out))
