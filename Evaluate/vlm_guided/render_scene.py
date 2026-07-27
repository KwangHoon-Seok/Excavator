"""
VLM-guided GCRL 실험용 씬 렌더 — **학습(training) env**(장애물 + 단일 랜덤 goal)에서
(1) cabin_cam RGB, (2) 극좌표 grid map(장애물만). pile/truck 없음(loading용 VLM/make_gridmap.py와 분리).

이 실험 = "VLM 서브골이 obstacle-aware GCRL 성능을 끌어올리는가" 증명(A/B paired).
  A(baseline) = GCRL이 goal로 직접   /   B = GCRL + VLM 서브골(장애물 회피 waypoint)
동일 seed로 reset → 장애물·goal이 결정론적으로 재현되므로, 여기서 렌더한 씬 = 실제 실행 씬.

목표(goal)는 자유공간 waypoint라 이미지에 안 보임 → scene.json/VLM 지시문에 **좌표로 전달**.
장애물만 gridmap 높이 blob으로 표시(트럭·흙더미 없음).

  conda run -n seok python Evaluate/vlm_guided/render_scene.py --seed 3
산출: Evaluate/vlm_guided/out/seed3/  cabin_init.png · gridmap.png · gridmap.npz/json · gridmap_legend.txt · scene.json
"""
import os
import sys
import json
import argparse
import numpy as np
import mujoco
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                  # repo root
sys.path.insert(0, os.path.join(ROOT, "Model", "Version4"))   # ExcavatorRHEREnv, HOME, reward
sys.path.insert(0, os.path.join(ROOT, "VLM"))                  # gridmap
import gridmap                                                 # noqa: E402
from excavator_env import ExcavatorRHEREnv                     # noqa: E402

_CFG = yaml.safe_load(open(os.path.join(ROOT, "Model", "Version4", "config.yaml")))
# 학습과 동일한 env 구성(장애물 분포·배치 제약 일치) — 실행 env와 렌더 env가 같아야 씬이 유효
_ENV_KEYS = ("reward", "random_start", "normalize_goal", "goal_azimuth_limit_deg",
             "obstacle_training", "n_obstacles", "n_obs_geom",
             "obs_len_range", "obs_wid_range", "obs_hgt_range",
             "obs_start_margin", "obs_goal_margin", "obs_path_relevant", "obs_path_corridor",
             "obs_place_tries", "obs_path_tmax", "ttc_max", "w_ttc", "curriculum_mode", "az_start_deg",
             "eps", "max_steps", "frame_skip")


def build_env_from(cfg_path):
    """지정 config(yaml 경로)로 env 생성 — ID/OOD 분포를 명시적으로 지정할 때 사용."""
    import yaml as _yaml
    return build_env(_yaml.safe_load(open(cfg_path)))


def build_env(cfg=_CFG):
    """config.yaml 기준 학습 env 인스턴스(장애물 회피). A/B 실행 러너와 동일 구성."""
    kw = {k: cfg[k] for k in _ENV_KEYS if k in cfg}
    return ExcavatorRHEREnv(**kw)


def active_obstacles(env):
    """reset 후 활성 장애물 → [{body, center[x,y,z], half[hx,hy,hz]}].
    env._active_obs(활성 box geom id)로부터 body명/중심/half를 수집. gridmap object_bodies·scene.json용."""
    m = env.model
    out = []
    for og in env._active_obs:
        bid = int(m.geom_bodyid[og])
        out.append({"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid),
                    "center": [round(float(v), 3) for v in m.body_pos[bid]],
                    "half": [round(float(v), 3) for v in m.geom_size[og]]})
    return out


def render_current(env, out, seed, n_r=45, n_theta=90, r_max=4.5, w=1280, h=480, fovx=135.0):
    """이미 reset된 env의 현재 상태 → cabin RGB + 장애물 gridmap + scene.json을 out 폴더에 저장.
    렌더 씬 == 실행 씬(같은 env 인스턴스) 보장. 반환: scene dict(seed/goal/start_tip/obstacles/paths).
    mission_eval 러너가 재사용(장애물 env 렌더 단일 소스)."""
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    m, d = env.model, env.data
    goal = [round(float(v), 3) for v in env._desired]
    start_tip = [round(float(v), 3) for v in env._tip()]
    obstacles = active_obstacles(env)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cabin_cam")
    m.cam_fovy[cid] = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(fovx) / 2) / (w / h)))
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera="cabin_cam")
    rgb_path = os.path.join(out, "cabin_init.png")
    Image.fromarray(r.render()).save(rgb_path)
    r.close()

    grid = gridmap.polar_height_grid(m, d, n_r=n_r, n_theta=n_theta, r_max=r_max,
                                     object_bodies=[o["body"] for o in obstacles],
                                     ground_z=gridmap.GROUND_Z)
    grid_path = gridmap.render_grid(grid, os.path.join(out, "gridmap.png"), px=(w, h))
    gridmap.save_grid(grid, os.path.join(out, "gridmap.npz"), os.path.join(out, "gridmap.json"))
    open(os.path.join(out, "gridmap_legend.txt"), "w").write(gridmap.color_legend(r_max=r_max))

    scene = {"seed": int(seed), "goal": goal, "start_tip": start_tip, "obstacles": obstacles,
             "rgb": rgb_path, "grid": grid_path,
             "h_range": [float(grid["H"].min()), float(grid["H"].max())]}
    json.dump({k: scene[k] for k in ("seed", "goal", "start_tip", "obstacles")},
              open(os.path.join(out, "scene.json"), "w"), indent=2)
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="reset seed — 장애물·goal 결정론 재현 키")
    ap.add_argument("--out", default=None, help="출력 폴더(기본 out/seed{N})")
    ap.add_argument("--n_r", type=int, default=45)
    ap.add_argument("--n_theta", type=int, default=90)
    ap.add_argument("--r_max", type=float, default=4.5)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fovx", type=float, default=135.0, help="RGB 수평 FOV(도)")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "out", f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)

    env = build_env()
    env.reset(seed=a.seed)                              # 장애물 배치 + goal 샘플(결정론)
    scene = render_current(env, out, a.seed, n_r=a.n_r, n_theta=a.n_theta, r_max=a.r_max,
                           w=a.w, h=a.h, fovx=a.fovx)
    print(f"seed={a.seed}  goal={scene['goal']}  start_tip={scene['start_tip']}  "
          f"장애물={len(scene['obstacles'])}개  gridmap h=[{scene['h_range'][0]:.2f}, {scene['h_range'][1]:.2f}] m")
    print("saved RGB :", scene["rgb"])
    print("saved grid:", scene["grid"])
    print("saved scene:", os.path.join(out, "scene.json"))
    env.close()


if __name__ == "__main__":
    main()
