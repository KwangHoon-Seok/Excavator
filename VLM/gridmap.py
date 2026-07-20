"""
polar height grid map — 굴착기 기준 (θ, r) 극좌표 지형/물체 높이맵.

VLM 입력용 "color-height 3D grid map"의 생성기.
- 원점: 선회축 world (-0.25, 0). forward(θ=0) = +y (swing=0에서 팔 헤딩).
- 커버: 전방 반원 θ∈[-90°,+90°] (θ>0 = 우측 +x), r∈[0, r_max].
- 높이(color): 평탄지면(z=-0.445) 기준 **부호 있는** 높이 h = z - ground_z.
    h<0 = 파인 곳(dig), h=0 = 평탄, h>0 = 쌓임/물체(rock 등).
- 높이원: hfield 지형(bilinear 샘플) + object body(free-joint 또는 지정) 상단을 셀별 max.

produces:
- polar_height_grid(model, data, ...) → dict(H, theta_c, r_c, X, Y, ground_z, ...)
- render_grid(grid, path)            → 직사각형(θ×r) heatmap PNG (VLM 입력용, 5밴드 명확색)
- save_grid(grid, npy_path)          → raw θ×r 높이배열 .npy + 메타 .json
"""
import json
import numpy as np
import mujoco

GROUND_Z = 0.0             # 평탄 지면 표면 z (excavator.xml: plane 지면, hfield 제거됨)
# 극좌표 원점 = world (0,0). VLM의 x=r·sinθ, y=r·cosθ 가 곧 world x,y (로컬=월드).
# (실제 선회축은 world (-0.25,0)이나 0.25m 차이는 무시 — world 정합 우선)
SWING_ORIGIN = (0.0, 0.0)


def _sample_terrain_z(model, X, Y, ground_z=GROUND_Z):
    """지면은 평탄 plane(z=ground_z) → 어디서나 ground_z. (hfield 제거로 지형 기복 없음.
    장애물/물체 높이는 polar_height_grid가 geom AABB로 별도 덮어씀.)"""
    return np.full_like(X, ground_z, dtype=float)


def _object_geoms(model, data, object_bodies=None):
    """물체로 볼 geom id 목록. object_bodies 지정 시 그 body 하위, 없으면 free-joint body 하위."""
    if object_bodies is not None:
        bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in object_bodies]
        bids = [b for b in bids if b >= 0]
    else:
        bids = []
        for b in range(model.nbody):
            ja = model.body_jntadr[b]
            if model.body_jntnum[b] >= 1 and model.jnt_type[ja] == mujoco.mjtJoint.mjJNT_FREE:
                bids.append(b)
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bids]
    return gids


def polar_height_grid(model, data, *, origin=SWING_ORIGIN, r_max=4.5, r_min=0.0,
                      n_r=45, n_theta=90, half_span_deg=90.0,
                      object_bodies=None, ground_z=GROUND_Z):
    """전방 반원 극좌표 높이맵을 계산. forward(θ=0)=+y, θ>0=우측(+x).

    반환 dict:
      H       : (n_theta, n_r) 부호 높이 h=z-ground_z (아래로 팜=음수, 물체=양수)
      z       : (n_theta, n_r) world z
      theta_c : (n_theta,) 셀중심 각도[rad]  r_c: (n_r,) 셀중심 반경[m]
      X, Y    : (n_theta, n_r) 셀중심 world xy
      ground_z, origin, r_max, half_span_deg
    """
    half = np.deg2rad(half_span_deg)
    theta_e = np.linspace(-half, half, n_theta + 1)          # 각 edge
    r_e = np.linspace(r_min, r_max, n_r + 1)                  # 반경 edge
    theta_c = 0.5 * (theta_e[:-1] + theta_e[1:])
    r_c = 0.5 * (r_e[:-1] + r_e[1:])
    TH, RR = np.meshgrid(theta_c, r_c, indexing="ij")        # (n_theta, n_r)
    # θ=0 → +y(forward), θ>0 → +x(우측):  x=ox+r sinθ, y=oy+r cosθ
    X = origin[0] + RR * np.sin(TH)
    Y = origin[1] + RR * np.cos(TH)

    z = _sample_terrain_z(model, X, Y, ground_z)             # 평탄 지면 표면 z (=ground_z)

    # object 상단을 셀별 max 로 덮어씀. footprint(xy)와 높이(z)를 분리:
    # geom 로컬 AABB 8코너를 월드로 변환 → 월드 축정렬 bbox(발자국) + 최상단 z.
    # (geom_rbound[바운딩구]는 높은 물체의 xy를 과대평가하므로 사용 안 함)
    gids = _object_geoms(model, data, object_bodies)
    for g in gids:
        gp = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3)
        did = model.geom_dataid[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and did >= 0:
            # mesh(트럭 등): 실제 정점을 월드로 → 발자국(xy AABB)+최상단 z.
            va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
            V = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3)
            pts = gp + V @ R.T
        else:
            # box: geom_size(런타임 갱신) 8코너. geom_aabb는 컴파일 시점값이라 미사용.
            hs = np.array(model.geom_size[g], float)
            pts = np.array([gp + R @ (hs * np.array([sx, sy, sz]))
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        top_z = float(pts[:, 2].max())                       # 물체 최상단 높이
        mask = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
        if mask.any():
            z[mask] = np.maximum(z[mask], top_z)

    H = z - ground_z
    return dict(H=H.astype(np.float32), z=z.astype(np.float32),
                theta_c=theta_c, r_c=r_c, theta_e=theta_e, r_e=r_e,
                X=X, Y=Y, ground_z=float(ground_z), origin=tuple(origin),
                r_max=float(r_max), half_span_deg=float(half_span_deg),
                n_object_geoms=len(gids))


# ── 높이 밴드 + 색 (프롬프트로 VLM에 알려줄 값) ──────────────────
#   높이 h = surface_z − ground(0). BANDS = 밴드 경계(n+1개) → n밴드.
#   지면 plane(hfield/dig 제거) → 음수 없음. 장애물 박스 높이 0.7~2.8 m 범위.
#   지면(0~0.4)=회색 중립, 그 위 장애물은 노랑→주황→빨강 램프(높을수록 붉음, YlOrRd).
BANDS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
BAND_COLORS = ["#d9d9d9", "#ffffb2", "#fed976", "#feb24c",
               "#fd8d3c", "#fc4e2a", "#e31a1c"]
BAND_DESC = ["지면 0~0.4", "0.4~0.8", "0.8~1.2", "1.2~1.6",
             "1.6~2.0", "2.0~2.4", "2.4~2.8 m (장애물)"]


def _cmap_norm(bands=BANDS, colors=BAND_COLORS):
    """명시적 밴드 경계 + 밴드별 명확한 색 → ListedColormap + BoundaryNorm."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    lc = ListedColormap(colors)
    return lc, BoundaryNorm(bands, lc.N, clip=True)      # 범위 밖은 양끝 밴드로 clip


def render_grid(grid, path, px=(1280, 480), dpi=100, bands=BANDS, colors=BAND_COLORS):
    """직사각형 언랩(θ×r) heatmap PNG — 프레임을 꽉 채우고 방위각/거리 축 라벨 표시.

    - 가로축 = 방위각 θ (−90°=왼쪽 … 0°=정면 … +90°=오른쪽)
    - 세로축 = 거리 r  (아래 0 … 위 r_max m)
    - 색 = 지면대비 높이 h. 밴드별 명확한 이산색(굴착=파랑/지면=회색/높이=초록·주황·빨강).
    - px: 출력 픽셀(기본 1280x480 = RGB와 동일).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.patches import Patch
    lc, norm = _cmap_norm(bands, colors)
    theta_deg = np.rad2deg(grid["theta_e"])                # (n_theta+1,)
    r_e = grid["r_e"]                                      # (n_r+1,)
    half = grid["half_span_deg"]; r_max = grid["r_max"]

    fig = plt.figure(figsize=(px[0] / dpi, px[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    # C=(n_r, n_theta): 행=거리, 열=방위각. shading="flat"로 edge 격자와 정합.
    # mm 반올림: 평탄지면 h=0의 부동소수점 미세오차(→굴착밴드 오분류) 제거.
    Hplot = np.round(grid["H"], 3)
    ax.pcolormesh(theta_deg, r_e, Hplot.T, cmap=lc, norm=norm, shading="flat")
    # 기존 형식 유지: 데이터(0~r_max)가 프레임을 꽉 채움
    ax.set_xlim(-half, half); ax.set_ylim(0, r_max)
    ax.set_xticks(range(-90, 91, 15))
    ax.set_yticks(np.arange(0, r_max + 0.01, 0.5))
    ax.tick_params(labelsize=11, length=3)
    ax.grid(True, color="0.55", lw=0.5, alpha=0.6)
    ax.axvline(0, color="0.35", lw=1.0, ls="--")           # θ=0 정면(forward) 기준선
    ax.set_xlabel("azimuth  θ (deg)   [−90 left  |  0 forward  |  +90 right]", fontsize=11)
    ax.set_ylabel("distance  r (m)", fontsize=11)
    # 트럭 포함 모든 물체는 높이 blob으로 표시 — VLM이 RGB(외형)↔gridmap(위치)을 대조해 종류 판단.

    # 범례: 오른쪽 상단에 오버레이(상단 고-r 영역은 대개 빈 평지). 색↔높이 밴드.
    labels = [f"{bands[i]:g}~{bands[i+1]:g}" for i in range(len(colors))]
    labels[0] += " (ground)"
    handles = [Patch(facecolor=colors[i], edgecolor="0.3", label=labels[i])
               for i in range(len(colors))]
    ax.legend(handles=handles, loc="lower right", ncol=5, fontsize=9,
              title="height h (m)", title_fontsize=9, framealpha=0.92,
              borderpad=0.4, handlelength=1.1, handleheight=1.1,
              columnspacing=0.9, handletextpad=0.4).set_zorder(10)
    fig.subplots_adjust(left=0.045, right=0.997, top=0.99, bottom=0.13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# 하위호환 별칭(기존 호출부 대비)
render_polar = render_grid


def color_legend(bands=BANDS, colors=BAND_COLORS, desc=BAND_DESC, r_max=4.5):
    """프롬프트용 색↔높이 매핑 설명 텍스트(사람이 읽고 프롬프트에 삽입)."""
    lines = [
        f"3D grid map = 굴착기 기준 (방위각 θ, 거리 r) 직사각형 높이맵(top-down 언랩). "
        f"가로축=방위각 θ: 왼쪽 −90° … 중앙 0°(정면/forward) … 오른쪽 +90°.  "
        f"세로축=거리 r: 아래 0 … 위 {r_max} m.  (셀 = 굴착기 기준 극좌표 위치)",
        "색 = 지면 대비 높이 h (m). 밴드별 명확한 색(범위 밖은 양끝 밴드로 clip):",
    ]
    for i, col in enumerate(colors):
        d = desc[i] if i < len(desc) else ""
        lines.append(f"  h {bands[i]:+.1f} ~ {bands[i+1]:+.1f} m : {col}   {d}")
    return "\n".join(lines)


def save_grid(grid, npy_path, meta_path=None):
    """raw θ×r 높이배열 + 좌표축을 .npy(.npz)로, 메타를 .json으로 저장."""
    np.savez(npy_path, H=grid["H"], z=grid["z"],
             theta_c=grid["theta_c"], r_c=grid["r_c"], X=grid["X"], Y=grid["Y"])
    if meta_path:
        meta = {k: grid[k] for k in ("ground_z", "origin", "r_max", "half_span_deg", "n_object_geoms")}
        meta["shape"] = list(grid["H"].shape)
        json.dump(meta, open(meta_path, "w"), indent=2)
    return npy_path
