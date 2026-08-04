"""
waypoint 정책 시각화: 뷰어 롤아웃(목표=초록 구) 또는 offscreen GIF. 목표는 매 에피소드 랜덤.
실행:
  뷰어:  conda run -n seok python Evaluate/visualize.py --ckpt Model/checkpoints/<run>.zip
  GIF :  conda run -n seok RENDER=gif python Evaluate/visualize.py --ckpt ...
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
                        np.array([0.1, 0.9, 0.1, 0.8], np.float32))
    scn.ngeom += 1


def run_viewer(model, env, episodes=5):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        for ep in range(episodes):
            obs, _ = env.reset(); done = False; info = {}
            while v.is_running() and not done:
                t0 = time.time()
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                v.user_scn.ngeom = 0; goal_geom(v.user_scn, env._desired); v.sync()
                done = term or trunc
                dt = env.frame_skip * env.model.opt.timestep - (time.time() - t0)
                if dt > 0: time.sleep(dt)
            print(f"ep{ep}: success={info.get('is_success')} dist={info.get('dist'):.3f}")
            if not v.is_running(): break
            # 도달한 마지막 자세를 잠깐 보여주고 다음 에피소드로
            t_hold = time.time()
            while v.is_running() and time.time() - t_hold < 1.5:
                v.sync(); time.sleep(1 / 60)


def run_gif(model, env, out, episodes=1):
    from PIL import Image
    r = mujoco.Renderer(env.model, 540, 720)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 1.6, 0.3]; cam.distance = 6.5; cam.azimuth = 135; cam.elevation = -22
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset(); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rw, term, trunc, info = env.step(a)
            r.update_scene(env.data, cam); goal_geom(r.scene, env._desired)
            frames.append(Image.fromarray(r.render())); done = term or trunc
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0)
    print(f"GIF 저장: {out} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ep", type=int, default=5)
    a = ap.parse_args()
    env = ExcavatorRHEREnv()
    model = SAC.load(a.ckpt, env=env, device="cpu")
    if os.environ.get("RENDER") == "gif":
        run_gif(model, env, os.path.join(ROOT, "Evaluate", "plots", "rollout_waypoint.gif"), a.ep)
    else:
        run_viewer(model, env, a.ep)


if __name__ == "__main__":
    main()
