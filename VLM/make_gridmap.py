"""
VLM 입력 페어 생성 데모/CLI — 같은 씬에서 (1) cabin_cam RGB, (2) 극좌표 3D grid map.
둘 다 480x1280(동일 종횡비)로 저장 → VLM에 나란히 입력.

  conda run -n seok python VLM/make_gridmap.py            # 데모(rock 주입+굴착)
  conda run -n seok python VLM/make_gridmap.py --plain    # 순수 excavator.xml 현상태

산출: VLM/out/  cabin_init.png (RGB) · gridmap.png (높이맵) · gridmap.npz/json · gridmap_legend.txt
색↔높이 매핑(colorbar/범위)은 이미지에 넣지 않고 legend 텍스트로 출력(→ 프롬프트에 삽입).
"""
import os, sys, argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))   # .../VLM
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)     # terrain (공유 자산)
sys.path.insert(0, HERE)     # gridmap (VLM 모듈)
import terrain
import gridmap

XML = os.path.join(ROOT, "excavator.xml")
OUT = os.path.join(HERE, "out")

import scene   # 공유 데모 씬(트럭+장애물+dig) — sequential.py와 동일 씬


def build_model(with_rocks):
    return scene.build_model() if with_rocks else mujoco.MjModel.from_xml_path(XML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true", help="rock 주입/굴착 없이 현재 xml 그대로")
    ap.add_argument("--n_r", type=int, default=45)
    ap.add_argument("--n_theta", type=int, default=90)
    ap.add_argument("--r_max", type=float, default=4.5)
    ap.add_argument("--w", type=int, default=1280)     # RGB/grid 공통 픽셀 크기
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--swing", type=float, default=0.0, help="선회각(rad) — 카메라도 함께 회전")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    m = build_model(with_rocks=not a.plain)
    d = mujoco.MjData(m)
    terrain.reset_terrain(m)                       # hfield 평탄 초기화(필수)
    d.qpos[0] = a.swing

    if not a.plain:
        scene.carve_dig(m)                         # 왼쪽 굴착지(공유 씬) — 트럭은 build서 안착됨
    mujoco.mj_forward(m, d)

    # (1) cabin_cam RGB (grid와 동일 픽셀 크기)
    from PIL import Image
    r = mujoco.Renderer(m, a.h, a.w)
    r.update_scene(d, camera="cabin_cam")
    rgb_path = os.path.join(OUT, "cabin_init.png")
    Image.fromarray(r.render()).save(rgb_path)
    r.close()

    # (2) 극좌표 3D grid map (동일 씬)
    obj_bodies = None if a.plain else scene.SCENE_BODIES   # 정적 물체 → 명시 지정으로 인식
    grid = gridmap.polar_height_grid(m, d, n_r=a.n_r, n_theta=a.n_theta, r_max=a.r_max,
                                     object_bodies=obj_bodies)
    print(f"grid H shape={grid['H'].shape}  object_geoms={grid['n_object_geoms']}  "
          f"h range=[{grid['H'].min():.3f}, {grid['H'].max():.3f}] m")
    grid_path = gridmap.render_grid(grid, os.path.join(OUT, "gridmap.png"), px=(a.w, a.h))
    gridmap.save_grid(grid, os.path.join(OUT, "gridmap.npz"), os.path.join(OUT, "gridmap.json"))

    # (3) 색↔높이 legend (프롬프트용) → 콘솔 + 파일
    legend = gridmap.color_legend(r_max=a.r_max)
    leg_path = os.path.join(OUT, "gridmap_legend.txt")
    open(leg_path, "w").write(legend)

    print("saved RGB :", rgb_path)
    print("saved grid:", grid_path, "(+ .npz/.json)")
    print("saved legend:", leg_path)
    print("\n===== 프롬프트용 색 legend =====\n" + legend)


if __name__ == "__main__":
    main()
