"""
사용자가 waypoint(목표점)를 지정 → 학습된 정책으로 bucket_tip을 거기로 보내는 추론 스크립트.

예:
  # 특정 waypoint로 (x y z, base/world 좌표 m)
  conda run -n seok python Evaluate/infer_waypoint.py --ckpt Model/checkpoints/single_waypoint.zip --wp 0.3 3.0 0.1
  # waypoint 미지정 시 도달가능 랜덤점 사용
  conda run -n seok python Evaluate/infer_waypoint.py --ckpt Model/checkpoints/single_waypoint.zip
  # 뷰어로 보기 / GIF 저장
  conda run -n seok python Evaluate/infer_waypoint.py --ckpt ... --wp 0.3 3.0 0.1 --render viewer
  conda run -n seok python Evaluate/infer_waypoint.py --ckpt ... --wp 0.3 3.0 0.1 --render gif
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco
from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Model"))
from excavator_env import ExcavatorRHEREnv


def goal_geom(scn, pos):
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.25, 0, 0.]), np.asarray(pos, float), np.eye(3).flatten(),
                        np.array([0.1, 0.9, 0.1, 0.85], np.float32))
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "Model/checkpoints/single_waypoint.zip"))
    ap.add_argument("--wp", type=float, nargs=3, default=None, help="waypoint x y z (미지정 시 랜덤 도달점)")
    ap.add_argument("--render", choices=["none", "gif", "viewer"], default="none")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # waypoint task: home 자세에서 시작, --wp 지정 시 options={"goal":...}로 목표 지정(없으면 랜덤).
    env = ExcavatorRHEREnv(max_steps=a.max_steps)
    model = SAC.load(a.ckpt, env=env, device="cpu")

    opts = {"goal": a.wp} if a.wp is not None else None
    obs, _ = env.reset(seed=a.seed, options=opts)
    wp = env._desired.copy()
    print(f"waypoint = {np.round(wp, 3)} | 시작 bucket_tip = {np.round(env._tip(), 3)} "
          f"| 초기거리 = {np.linalg.norm(env._tip() - wp):.3f} m")

    # ── 추론 루프 ──
    if a.render == "viewer":
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            done = False; info = {}
            while v.is_running() and not done:
                t0 = time.time()
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                v.user_scn.ngeom = 0; goal_geom(v.user_scn, env._desired); v.sync()
                done = term or trunc
                dt = env.frame_skip * env.model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            print(f"결과: success={info.get('is_success')} 최종거리={info.get('dist'):.3f} m  "
                  f"최종 bucket_tip={np.round(env._tip(), 3)}")
            print("창을 닫으면 종료됩니다. (목표=초록 구)")
            while v.is_running():           # 도달 자세 유지 — 사용자가 닫을 때까지
                v.user_scn.ngeom = 0; goal_geom(v.user_scn, env._desired); v.sync()
                time.sleep(1 / 60)
        return

    frames = []
    if a.render == "gif":
        from PIL import Image
        r = mujoco.Renderer(env.model, 540, 720)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 1.6, 0.5]; cam.distance = 6.5; cam.azimuth = 135; cam.elevation = -20

        def shot():
            r.update_scene(env.data, cam); goal_geom(r.scene, env._desired)
            return Image.fromarray(r.render())

    done = False; info = {}
    if a.render == "gif":
        frames.append(shot())
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, rw, term, trunc, info = env.step(act)
        if a.render == "gif":
            frames.append(shot())
        done = term or trunc
    print(f"결과: success={info.get('is_success')} 최종거리={info.get('dist'):.3f} m  "
          f"최종 bucket_tip={np.round(env._tip(), 3)}")

    if a.render == "gif":
        out = os.path.join(ROOT, "Evaluate/plots", "infer_waypoint.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0)
        print(f"GIF 저장: {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
