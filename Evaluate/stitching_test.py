"""
순수 stitching 성능 검증 — compounding 제거.
각 시행 독립: [시작 자세] → [단일 goal] 도달만 측정(연쇄 X).
- home 시작 세트 vs 랜덤 시작 세트를 **goal 거리 분포(1.5~3.0m) 동일하게** 맞춰 비교
  → 두 세트의 차이는 오직 '시작 자세(home vs 임의)'. = 순수 stitching.
- 동일한 (시작자세, goal) 쌍을 두 모델에 똑같이 먹임(고정 seed).
순수 stitching 갭 = (home 시작 성공률) − (랜덤 시작 성공률). 작을수록 stitching 강함.

실행: conda run -n seok python Evaluate/stitching_test.py
"""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.join("/home/aimlab/qwen-vlm", "Model"))
from excavator_env import ExcavatorRHEREnv, HOME
from stable_baselines3 import SAC

ROOT = "/home/aimlab/qwen-vlm"
MODELS = {
    "random_O (Total_reward)": f"{ROOT}/Model/checkpoints/Version2/Total_reward_0705.zip",
    "random_X (without_rand)": f"{ROOT}/Model/checkpoints/Version2/without_random_position_0706.zip",
}
N = 200             # 세트당 시행 수
D_LO, D_HI = 1.5, 3.0   # 시작 tip → goal 거리 밴드(난이도 통제)
MAX_STEPS = 300


def make_env():
    return ExcavatorRHEREnv(random_start=False, normalize_goal=True,
                            goal_azimuth_limit_deg=120, eps=0.2, max_steps=MAX_STEPS)


def gen_pairs(env, start_mode):
    """(start_qpos, goal) 쌍 N개 생성 — goal은 시작 tip에서 D_LO~D_HI m 떨어진 도달점."""
    pairs = []
    for _ in range(N):
        if start_mode == "home":
            qpos = HOME.copy()
        else:
            qpos = env.np_random.uniform(env.ctrl_lo, env.ctrl_hi)   # 학습 random_start와 동일 분포
        env.data.qpos[:4] = qpos; mujoco.mj_forward(env.model, env.data)
        tip = env._tip()
        goal = None
        for _ in range(300):                                          # 거리밴드 안 도달 goal 샘플
            idx = int(env.np_random.integers(0, env._cell_r.size))
            r, z = env._cell_r[idx], env._cell_z[idx]
            th = float(env.np_random.uniform(-np.pi, np.pi))
            g = np.array([r * np.cos(th), r * np.sin(th), z])
            if D_LO <= np.linalg.norm(g - tip) <= D_HI:
                goal = g; break
        if goal is None:
            goal = g
        pairs.append((qpos.copy(), goal))
    return pairs


def run_trial(model, env, qpos, goal):
    """독립 시행: 시작 자세 세팅 → goal 도달 시도(리셋으로 지형만 초기화)."""
    env.reset(seed=0)                              # 지형 초기화(시작자세·goal은 아래서 덮어씀)
    env.data.qpos[:4] = qpos; env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._desired = np.asarray(goal, float)
    env.data.ctrl[:] = env.data.qpos[:4]
    env._prev_action = np.zeros(4, np.float32); env._steps = 0
    obs = env._get_obs(); info = {}
    for _ in range(MAX_STEPS):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, tr, info = env.step(a)
        if info.get("is_success"):
            return True
    return False


def eval_set(model, env, pairs):
    return np.mean([run_trial(model, env, q, g) for q, g in pairs])


def main():
    genv = make_env(); genv.reset(seed=2024)      # 쌍 생성용(고정 seed → 두 모델 동일 쌍)
    home_pairs = gen_pairs(genv, "home")
    rand_pairs = gen_pairs(genv, "random")
    # 실제 시작거리 평균(난이도 통제 확인)
    dh = np.mean([np.linalg.norm(_tip_of(q) - g) for q, g in home_pairs])
    dr = np.mean([np.linalg.norm(_tip_of(q) - g) for q, g in rand_pairs])
    print(f"시행 {N}/세트, goal거리밴드 {D_LO}~{D_HI}m  (실제 평균: home {dh:.2f}m / random {dr:.2f}m)\n")
    print(f"{'모델':26s} | {'home시작':>9s} | {'랜덤시작':>9s} | {'순수 stitching 갭':>16s}")
    print("-" * 74)
    for name, ck in MODELS.items():
        env = make_env(); model = SAC.load(ck, env=env, device="cpu")
        sh = eval_set(model, env, home_pairs)
        sr = eval_set(model, env, rand_pairs)
        print(f"{name:26s} | {sh:9.2f} | {sr:9.2f} | {sh - sr:+16.2f}")
    print("\n갭 작을수록 stitching 강함(임의 시작에서도 잘 도달). 큰 갭 = home 편향(비-home 시작 취약).")


_GENV = None
def _tip_of(qpos):
    global _GENV
    if _GENV is None:
        _GENV = make_env(); _GENV.reset(seed=0)
    _GENV.data.qpos[:4] = qpos; mujoco.mj_forward(_GENV.model, _GENV.data)
    return _GENV._tip()


if __name__ == "__main__":
    main()
