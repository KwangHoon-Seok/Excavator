"""
Multi-goal sequential 실행 테스트: 학습 정책이 waypoint 시퀀스를 순서대로 도달하는지.
- 시작 = home 고정. 각 goal 도달(‖tip−goal‖<eps) 시 **리셋 없이** 다음 goal로 전환(상태 연속).
- random_start로 "임의 시작→도달"을 배웠으니, 시퀀스 = 그 단일도달의 연쇄(stitching) 검증.

실행:
  chained(자동, in-distribution 시퀀스):
    conda run -n seok python Evaluate/sequential.py --ckpt Model/checkpoints/<run>.zip --k 4
  explicit:
    conda run -n seok python Evaluate/sequential.py --ckpt ... --goals "0.3,3,0.1 -0.5,2.5,0.3 1,2,0.5"
  시각화:  --render gif  |  --render viewer
⚠️ --normalize_goal, --eps, --azimuth 는 학습 때 설정과 맞춰야 obs가 일치함(기본=현 config).
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer  # 최상단 import (main 안에서 import 시 mujoco가 지역변수化 → UnboundLocalError)
from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Model"))
sys.path.insert(0, ROOT)
from excavator_env import ExcavatorRHEREnv  # noqa: E402
import scene  # noqa: E402  # 공유 데모 씬(트럭+장애물+dig) — make_gridmap.py와 동일


def _sphere(scn, pos, rgba, size=0.2):
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0.]), np.asarray(pos, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_goals(scn, goals, cur):
    """도달=회색, 현재타깃=초록, 대기=빨강."""
    scn.ngeom = 0
    for j, g in enumerate(goals):
        rgba = ([0.6, 0.6, 0.6, 0.45] if j < cur else
                [0.1, 0.9, 0.1, 0.9] if j == cur else
                [0.9, 0.2, 0.2, 0.45])
        _sphere(scn, g, rgba)


def sample_chain(env, k):
    """home tip에서 시작, 각 goal을 '직전 goal(예상 tip) 전방 ±azimuth'에서 순차 샘플(in-dist).
    ⚠️ 방위창이 직전 goal 기준이라 chain이 drift → 카메라 시야 밖으로 벗어날 수 있음."""
    env.curr_r = None                       # 전체 범위(방위 제한은 center 기준 상대)
    goals, tip = [], env._tip()
    for _ in range(k):
        g = env._sample_reachable_waypoint(center=tip)
        goals.append(g.copy()); tip = g
    return goals


def sample_fov(env, k, half_fov_deg=60.0):
    """카메라 전방 시야(+y 기준 ±half_fov) 안의 도달가능 waypoint를 랜덤 k개.
    방위를 '고정된 전방'에 묶어 drift 없이 항상 카메라 화각 안에 있게 함."""
    half = np.radians(half_fov_deg)
    goals = []
    for _ in range(k):
        idx = int(env.np_random.integers(0, env._cell_r.size))
        r, z = env._cell_r[idx], env._cell_z[idx]        # 도달가능 (반경, 높이)
        phi = float(env.np_random.uniform(-half, half))  # +y(전방) 기준 방위(±half_fov)
        goals.append(np.array([r * np.sin(phi), r * np.cos(phi), z]))  # x=r·sinφ, y=r·cosφ
    return goals


def execute(model, env, goals, per_goal_steps, on_step=None):
    """goal 순차 실행 — 리셋 없이 _desired만 교체. (i, 도달여부, 스텝, dist) 리스트 반환."""
    results = []
    for i, g in enumerate(goals):
        env._desired = np.asarray(g, dtype=float)
        obs = env._get_obs()                # 새 goal 반영한 관측
        steps, reached, info = 0, False, {}
        while steps < per_goal_steps:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(act); steps += 1
            if on_step is not None:
                on_step(i)
            if info.get("is_success"):
                reached = True; break
        results.append((i, reached, steps, float(info.get("dist", np.nan))))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=4, help="chained goal 개수")
    ap.add_argument("--goals", help='explicit "x,y,z x,y,z ..." (주면 chained 무시)')
    ap.add_argument("--per_goal_steps", type=int, default=200)
    ap.add_argument("--fov", type=float, default=60.0, help="자동 goal을 전방 ±fov° 카메라 시야 안에서 샘플")
    ap.add_argument("--chained", action="store_true", help="자동 goal을 chained(drift 가능)로 샘플(구방식)")
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--normalize_goal", action="store_true")
    ap.add_argument("--azimuth", type=float, default=120.0)
    ap.add_argument("--render", choices=["none", "gif", "viewer"], default="none")
    ap.add_argument("--demo_scene", action="store_true",
                    help="트럭+장애물+dig 공유 씬에서 실행/시각화(make_gridmap과 동일 씬)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = ExcavatorRHEREnv(random_start=False, normalize_goal=a.normalize_goal,
                           goal_azimuth_limit_deg=a.azimuth, eps=a.eps, max_steps=10**9)
    if a.demo_scene:                        # 학습 env는 그대로, 실행용 모델만 씬으로 교체
        env.model = scene.build_model()     # (arm/센서/액추에이터 규격 동일 → obs·정책 호환)
        env.data = mujoco.MjData(env.model)
    model = SAC.load(a.ckpt, env=env, device="cpu")
    env.reset(seed=a.seed)                  # home 시작(random_start=False)
    if a.demo_scene:
        scene.carve_dig(env.model)          # reset가 지형 평탄화하므로 그 뒤 dig 재적용
        mujoco.mj_forward(env.model, env.data)

    if a.goals:
        goals = [np.array([float(v) for v in g.split(",")]) for g in a.goals.split()]
    elif a.chained:
        goals = sample_chain(env, a.k)      # 구방식(방위 drift 가능)
    else:
        goals = sample_fov(env, a.k, a.fov)  # 기본: 카메라 전방 ±fov° 시야 안
    print(f"시퀀스 {len(goals)}개 | 시작 home tip={np.round(env._tip(),2)}")

    # ── 실행 + 렌더 ──
    if a.render == "gif":
        from PIL import Image
        r = mujoco.Renderer(env.model, 540, 720)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 1.0, 0.3]; cam.distance = 7.0; cam.azimuth = 135; cam.elevation = -20
        frames = []

        def on_step(cur):
            r.update_scene(env.data, cam); draw_goals(r.scene, goals, cur)
            frames.append(Image.fromarray(r.render()))
        results = execute(model, env, goals, a.per_goal_steps, on_step)
        out = os.path.join(ROOT, "Evaluate", "plots", "sequential.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
        print(f"GIF 저장: {out} ({len(frames)} frames)")
    elif a.render == "viewer":
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            dt = env.frame_skip * env.model.opt.timestep

            def on_step(cur):
                v.user_scn.ngeom = 0; draw_goals(v.user_scn, goals, cur); v.sync(); time.sleep(dt)
            results = execute(model, env, goals, a.per_goal_steps, on_step)
            print("창을 닫으면 종료")
            while v.is_running():
                v.user_scn.ngeom = 0; draw_goals(v.user_scn, goals, len(goals)); v.sync(); time.sleep(1 / 60)
    else:
        results = execute(model, env, goals, a.per_goal_steps)

    # ── 결과 ──
    nok = sum(r[1] for r in results)
    for i, reached, steps, dist in results:
        print(f"  goal{i}: {'✅도달' if reached else '❌실패'}  steps={steps}  dist={dist:.3f}m")
    print(f"시퀀스 성공: {nok}/{len(goals)}  전체성공={'YES' if nok == len(goals) else 'NO'}")


if __name__ == "__main__":
    main()
