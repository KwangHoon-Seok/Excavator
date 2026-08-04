"""
정량 평가: 저장된 waypoint 정책을 N에피소드(랜덤 목표) 결정론 평가 → 성공률/평균거리.
실행: conda run -n seok python Evaluate/evaluate.py --ckpt Model/checkpoints/<run>.zip [--n 30]
"""
import os
import sys
import argparse
import numpy as np
from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Model"))
from excavator_env import ExcavatorRHEREnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    env = ExcavatorRHEREnv()
    model = SAC.load(a.ckpt, env=env, device="cpu")  # HER 모델은 env 필요

    succ, dists = 0, []
    for i in range(a.n):
        obs, _ = env.reset(seed=20000 + i)
        done = False; info = {}
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act); done = term or trunc
        succ += int(info.get("is_success", False)); dists.append(info.get("dist", np.nan))

    print(f"평가: {a.ckpt}  (n={a.n})")
    print(f"waypoint  success={succ / a.n * 100:.0f}%  mean_dist={np.nanmean(dists):.3f} m")


if __name__ == "__main__":
    main()
