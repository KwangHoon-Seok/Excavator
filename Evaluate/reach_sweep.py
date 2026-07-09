"""
굴착기 팔의 도달영역 정밀 측정 + 스윕 GIF.
- swing=0 고정(높이는 swing 무관). boom×stick×bucket 그리드 스윕 FK로 bucket_tip (r,z) 수집.
- r 구간별 z_max/z_min 표 출력(→ VLM 프롬프트에 명시).
- 스윕 과정을 측면뷰 GIF로 저장(팔 이동 + tip 궤적 누적 = 도달영역 시각화).
실행: conda run -n seok python Evaluate/reach_sweep.py
산출: Evaluate/plots/reach_sweep.gif  + 콘솔 표
"""
import os
import numpy as np
import mujoco
from PIL import Image

ROOT = "/home/aimlab/qwen-vlm"
XML = os.path.join(ROOT, "excavator.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
tip_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "bucket_tip_pos")]
BL, BH = m.actuator_ctrlrange[1]      # boom
SL, SH = m.actuator_ctrlrange[2]      # stick
KL, KH = m.actuator_ctrlrange[3]      # bucket


def tip():
    return d.sensordata[tip_adr:tip_adr + 3].copy()


# ── 1. 정밀 측정 (그리드 스윕 FK) ──
rs, zs = [], []
for bo in np.linspace(BL, BH, 22):
    for st in np.linspace(SL, SH, 32):
        for bk in np.linspace(KL, KH, 12):
            d.qpos[:4] = [0, bo, st, bk]
            mujoco.mj_forward(m, d)
            t = tip()
            rs.append(float(np.hypot(t[0], t[1]))); zs.append(float(t[2]))
rs, zs = np.array(rs), np.array(zs)

print("=== r별 도달 z 범위 (goal 영역 r∈[1.0,4.5]) ===")
print(f"{'r 구간(m)':12s} {'z_max(m)':>9s} {'z_min(m)':>9s}")
table = []
for lo in np.arange(1.0, 4.5, 0.5):
    sel = (rs >= lo) & (rs < lo + 0.5)
    if sel.any():
        zmax, zmin = zs[sel].max(), zs[sel].min()
        table.append((lo, lo + 0.5, zmax, zmin))
        print(f"  {lo:.1f}~{lo+0.5:.1f}      {zmax:+9.2f} {zmin:+9.2f}")
print(f"전체: r∈[{rs.min():.2f},{rs.max():.2f}]  z∈[{zs.min():.2f},{zs.max():.2f}]")

# ── 2. 스윕 GIF (측면뷰 + tip 궤적 누적) ──
r = mujoco.Renderer(m, 540, 720)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 2.0, 0.6]; cam.distance = 7.5; cam.azimuth = 25; cam.elevation = -12
frames, trail = [], []
BK_MID = 0.5 * (KL + KH)
for bo in np.linspace(BL, BH, 6):
    for st in np.linspace(SL, SH, 16):
        d.qpos[:4] = [0, bo, st, BK_MID]
        mujoco.mj_forward(m, d)
        trail.append(tip().copy())
        r.update_scene(d, cam)
        sc = r.scene
        for p in trail[::2]:                      # 궤적 = 반투명 초록 점(도달 tip)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.05, 0, 0.]), np.asarray(p, float),
                                np.eye(3).flatten(), np.array([0.1, 0.9, 0.1, 0.5], np.float32))
            sc.ngeom += 1
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,   # 현재 tip=빨강
                            np.array([0.11, 0, 0.]), np.asarray(trail[-1], float),
                            np.eye(3).flatten(), np.array([0.95, 0.1, 0.1, 1.0], np.float32))
        sc.ngeom += 1
        frames.append(Image.fromarray(r.render()))
out = os.path.join(ROOT, "Evaluate", "plots", "reach_sweep.gif")
frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0)
print(f"\nGIF 저장: {out} ({len(frames)} frames)")
