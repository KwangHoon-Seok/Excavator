"""
VLM 입력 페어 생성 — 상차(loading) 미션 씬(흙더미 + 덤프트럭 + 랜덤 장애물)에서
(1) cabin_cam RGB, (2) 극좌표 color-height grid map. 둘 다 동일 종횡비로 저장 → VLM에 나란히 입력.

★ 이 씬은 **평가(mission) 전용** — 흙더미·트럭 포함. (학습 env는 기존대로 장애물만, 변경 없음)
gridmap = 흙더미(source)+장애물(avoid) 높이. 트럭은 RGB로 인식(gridmap 제외).
scene.json에 장애물·pile_pick·truck_dump 저장 → mission_eval이 GCRL로 동일 장애물 재현·실행.

  conda run -n seok python VLM/make_gridmap.py --seed 3
산출: VLM/out/  cabin_init.png · gridmap.png · gridmap.npz/json · gridmap_legend.txt · scene.json
"""
import os, sys, json, argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Model", "Version4"))   # HOME 자세
sys.path.insert(0, HERE)                                       # gridmap, mission_scene
import gridmap
import mission_scene
from excavator_env import HOME

OUT = os.path.join(HERE, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="랜덤 장애물 시드(scene 재현용)")
    ap.add_argument("--n_r", type=int, default=45)
    ap.add_argument("--n_theta", type=int, default=90)
    ap.add_argument("--r_max", type=float, default=4.5)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fovx", type=float, default=135.0, help="RGB 수평 FOV(도)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    m, obstacles = mission_scene.build_model(a.seed)     # 흙더미+트럭 고정 + 장애물 랜덤
    d = mujoco.MjData(m)
    d.qpos[:4] = HOME                                    # 시작 자세(학습과 동일)
    mujoco.mj_forward(m, d)

    # (1) cabin_cam RGB (수평 FOV=fovx에 맞춰 fovy 설정)
    from PIL import Image
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cabin_cam")
    m.cam_fovy[cid] = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(a.fovx) / 2) / (a.w / a.h)))
    r = mujoco.Renderer(m, a.h, a.w)
    r.update_scene(d, camera="cabin_cam")
    rgb_path = os.path.join(OUT, "cabin_init.png")
    Image.fromarray(r.render()).save(rgb_path)
    r.close()

    # (2) 극좌표 grid map (흙더미 source + 장애물 avoid; 트럭 제외)
    grid = gridmap.polar_height_grid(m, d, n_r=a.n_r, n_theta=a.n_theta, r_max=a.r_max,
                                     object_bodies=mission_scene.gridmap_bodies(obstacles),
                                     ground_z=gridmap.GROUND_Z)
    print(f"grid H shape={grid['H'].shape}  물체 geom={grid['n_object_geoms']}  "
          f"h range=[{grid['H'].min():.2f}, {grid['H'].max():.2f}] m  장애물={len(obstacles)}개")
    grid_path = gridmap.render_grid(grid, os.path.join(OUT, "gridmap.png"), px=(a.w, a.h))
    gridmap.save_grid(grid, os.path.join(OUT, "gridmap.npz"), os.path.join(OUT, "gridmap.json"))

    # (3) legend + scene 저장(mission_eval/GCRL 재현용)
    open(os.path.join(OUT, "gridmap_legend.txt"), "w").write(gridmap.color_legend(r_max=a.r_max))
    json.dump({"seed": a.seed, "pile_pick": mission_scene.PILE_PICK, "truck_dump": mission_scene.TRUCK_DUMP,
               "obstacles": obstacles}, open(os.path.join(OUT, "scene.json"), "w"), indent=2)

    print("saved RGB :", rgb_path)
    print("saved grid:", grid_path)
    print("saved scene: %s/scene.json (pile_pick=%s truck_dump=%s 장애물 %d개)"
          % (OUT, mission_scene.PILE_PICK, mission_scene.TRUCK_DUMP, len(obstacles)))


if __name__ == "__main__":
    main()
