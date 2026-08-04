"""
평가 시나리오(미션 씬) 오버뷰 이미지 100장 생성 — 흙더미+트럭+장애물 전경 + goal 표시.
goal: 흙더미 픽업점(PILE_PICK)=초록 구, 트럭 투하점(TRUCK_DUMP)=파랑 구. 시작 tip(HOME)=회색 구.

  python Experiment/Code/render_scenarios.py --n 100 --dist id
산출: Experiment/Code/out/scenarios/<dist>/seed###.png
"""
import os
import sys
import argparse

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mission as M                                             # noqa: E402
sys.path.insert(0, os.path.join(M.ROOT, "Model", "Version4"))
from excavator_env import HOME                                  # noqa: E402

HOME_TIP = np.array([0.0, 3.62, 1.0])                           # 시작 버킷 tip 근사


def _sphere(scn, pos, rgba, size=0.18):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0.0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _marker_column(scn, pos, rgba, h=1.2):
    """goal 위치를 지면까지 잇는 얇은 기둥 + 상단 구(가시성)."""
    p = np.asarray(pos, float)
    base = p.copy(); base[2] = 0.02
    mid = (p + base) / 2
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([0.03, 0.03, np.linalg.norm(p - base) / 2]),
                            mid, np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
    _sphere(scn, p, rgba)


def render_seed(seed, out, dist, w=1000, h=750):
    obstacles = M.gen_obstacles(seed, dist)
    m = M.build_mission_model(obstacles)
    d = mujoco.MjData(m)
    d.qpos[:4] = HOME
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 2.0, 0.4]
    cam.distance = 9.5
    cam.azimuth = 90
    cam.elevation = -55
    r.update_scene(d, cam)
    _marker_column(r.scene, M.PILE_PICK, [0.10, 0.85, 0.10, 0.95])     # 흙더미 픽업 = 초록
    _marker_column(r.scene, M.TRUCK_DUMP, [0.20, 0.45, 1.0, 0.95])     # 트럭 투하 = 파랑
    _sphere(r.scene, HOME_TIP, [0.6, 0.6, 0.6, 0.9], size=0.14)        # 시작 tip = 회색
    from PIL import Image
    img = r.render(); r.close()
    Image.fromarray(img).save(out)
    return len(obstacles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds_file", default=os.path.join(HERE, "seeds_mission.json"))
    ap.add_argument("--dist", default="id", choices=["id", "ood"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import json
    if os.path.exists(a.seeds_file):
        seeds = [int(s) for s in json.load(open(a.seeds_file))][:a.n]
    else:
        seeds = list(range(a.n))
    out = a.out or os.path.join(HERE, "out", "scenarios", a.dist)
    os.makedirs(out, exist_ok=True)
    print(f"시나리오 {len(seeds)}장 렌더({a.dist}) → {out}")
    print(f"goal: 흙더미 픽업={M.PILE_PICK}(초록), 트럭 투하={M.TRUCK_DUMP}(파랑)")
    for i, s in enumerate(seeds):
        nobs = render_seed(s, os.path.join(out, f"seed{s:03d}.png"), a.dist)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(seeds)} (seed{s}, 장애물 {nobs}개)")
    print("완료:", out)


if __name__ == "__main__":
    main()
