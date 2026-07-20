"""유효 회피 시나리오 seed 선별 (논문 실험용) — 지정 분포 config + 고정 GCRL로 스크린.
기준: z≥zfloor(바닥충돌 제외) + free-space 도달(goal 실현) + 장애물 有시 A unsafe(실패/충돌).
→ genuine avoidance challenge만 남김. 구간별(dist×az) 구성도 리포트.

  conda run -n seok python Evaluate/vlm_guided/screen_seeds.py \\
    --config Model/checkpoints/GCRL_fixed_config.yaml --ckpt Model/checkpoints/GCRL_fixed.zip \\
    --start 700000 --scan 800 --need 60 --out Evaluate/vlm_guided/seeds_ID.json
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
import render_scene as RS
from stable_baselines3 import SAC


def bins(env, goal):
    s = env._tip_home; g = np.asarray(goal, float)
    fwd = np.arctan2(s[1], s[0]); dth = np.arctan2(g[1], g[0]) - fwd
    az = abs(np.degrees(np.arctan2(np.sin(dth), np.cos(dth))))
    d0 = float(np.linalg.norm(g - s))
    azb = "front" if az < 45 else ("side" if az < 90 else "wide")
    db = "near" if d0 < 2.0 else ("mid" if d0 < 3.5 else "far")
    return azb, db


def rollout(env, model, seed, free):
    obs, _ = env.reset(seed=seed)
    goal = np.asarray(env._desired, float).copy(); n_obs = len(env._active_obs)
    if free:
        for ob in env._obs_bodies:
            env.model.body_pos[ob] = [0, 0, 60.0]
        env._obs_sph = []; env._active_obs = []
        mujoco.mj_forward(env.model, env.data); obs = env._get_obs()
    collided, info, done = False, {}, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        if info.get("min_gap", 1e9) <= 0: collided = True
        done = term or trunc
    return goal, n_obs, bool(info.get("is_success", False)), collided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="분포 config(yaml). ID=GCRL_fixed_config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", type=int, default=700000)
    ap.add_argument("--scan", type=int, default=800)
    ap.add_argument("--need", type=int, default=60)
    ap.add_argument("--zfloor", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    env = RS.build_env_from(a.config); env.set_curriculum(1.0)
    model = SAC.load(a.ckpt, env=env, device="auto")
    stats = dict(lowz=0, infeasible=0, trivial=0, valid=0)
    valid, comp = [], {}
    for s in range(a.start, a.start + a.scan):
        goal, n_obs, ok_obs, coll_obs = rollout(env, model, s, free=False)
        if goal[2] < a.zfloor: stats["lowz"] += 1; continue
        if n_obs == 0: stats["trivial"] += 1; continue
        _, _, ok_free, _ = rollout(env, model, s, free=True)
        if not ok_free: stats["infeasible"] += 1; continue
        if (not ok_obs) or coll_obs:                 # A unsafe = genuine challenge
            stats["valid"] += 1
            azb, db = bins(env, goal)
            comp[f"{azb}/{db}"] = comp.get(f"{azb}/{db}", 0) + 1
            valid.append(dict(seed=s, gz=round(float(goal[2]), 2),
                              gr=round(float(np.hypot(goal[0], goal[1])), 1), n_obs=n_obs,
                              az=azb, dist=db, A_succ=int(ok_obs), A_coll=int(coll_obs)))
            if len(valid) >= a.need: break
        else:
            stats["trivial"] += 1

    json.dump([v["seed"] for v in valid], open(a.out, "w"))
    json.dump(valid, open(a.out.replace(".json", "_detail.json"), "w"), indent=2)
    print(f"스캔 {s - a.start + 1}개: {stats}")
    print(f"유효 {len(valid)}개 → {a.out}")
    print("구간 구성:", dict(sorted(comp.items())))


if __name__ == "__main__":
    main()
