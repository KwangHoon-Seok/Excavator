"""
두 정책을 sequential task로 공정 비교 — random_start(stitching)의 효과 검증.
- 동일한 goal 시퀀스(고정 seed, 월드좌표)를 두 모델에 똑같이 먹임.
- home 시작 → g1 → g2 → ... 리셋 없이 연쇄. g2+는 '직전 goal 위치(비-home)'에서 출발 = stitching 시험.
- goal 위치별(1번=home시작, 2번+=비-home시작) 성공률 → home고정 정책이 2번+에서 무너지는지 확인.

실행: conda run -n seok python Evaluate/compare_sequential.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join("/home/aimlab/qwen-vlm", "Model"))
from excavator_env import ExcavatorRHEREnv
from stable_baselines3 import SAC

ROOT = "/home/aimlab/qwen-vlm"
MODELS = {
    "random_O (Total_reward)": f"{ROOT}/Model/checkpoints/Version2/Total_reward_0705.zip",
    "random_X (without_rand)": f"{ROOT}/Model/checkpoints/Version2/without_random_position_0706.zip",
}
K_SEQ = 100         # 시퀀스 개수
L = 4               # 시퀀스당 goal 개수
PER_GOAL = 200      # goal당 최대 스텝
FOV = 60.0          # 전방 ±FOV° 안에서 goal 샘플


def make_env():
    return ExcavatorRHEREnv(random_start=False, normalize_goal=True,
                            goal_azimuth_limit_deg=120, eps=0.2, max_steps=10**9)


def gen_sequences():
    """월드좌표 goal 시퀀스 K개 생성(고정 seed → 두 모델에 동일하게 사용)."""
    env = make_env(); env.reset(seed=777)
    half = np.radians(FOV)
    seqs = []
    for _ in range(K_SEQ):
        seq = []
        for _ in range(L):
            idx = int(env.np_random.integers(0, env._cell_r.size))
            r, z = env._cell_r[idx], env._cell_z[idx]
            phi = float(env.np_random.uniform(-half, half))     # 전방 기준 ±FOV
            seq.append(np.array([r * np.sin(phi), r * np.cos(phi), z]))
        seqs.append(seq)
    return seqs


def run_model(ckpt, seqs):
    env = make_env()
    model = SAC.load(ckpt, env=env, device="cpu")
    # pos[i] = i번째 goal(0=home시작) 성공수, dist[i]=최종거리, start_d[i]=활성화 시 시작거리
    pos_succ = np.zeros(L); pos_dist = [[] for _ in range(L)]; start_d = [[] for _ in range(L)]
    full = 0
    for seq in seqs:
        env.reset(seed=0)                       # home 고정 시작
        all_ok = True
        for i, g in enumerate(seq):
            env._desired = np.asarray(g, float)
            start_d[i].append(float(np.linalg.norm(env._tip() - g)))   # 활성화 순간 tip-goal 거리
            obs = env._get_obs(); info = {}; ok = False
            for _ in range(PER_GOAL):
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, tr, info = env.step(a)
                if info.get("is_success"):
                    ok = True; break
            pos_succ[i] += ok; pos_dist[i].append(float(info.get("dist", np.nan)))
            all_ok = all_ok and ok
        full += all_ok
    return pos_succ / len(seqs), [np.nanmean(d) for d in pos_dist], \
        [np.mean(s) for s in start_d], full / len(seqs)


def main():
    seqs = gen_sequences()
    print(f"시퀀스 {len(seqs)}개 × goal {L}개, 전방±{FOV:.0f}°, per_goal={PER_GOAL}스텝\n")
    print(f"{'모델':26s} | {'goal별 성공률 (1=home시작, 2+=비-home)':40s} | 전체시퀀스")
    print("-" * 92)
    results = {}
    for name, ck in MODELS.items():
        ps, pd, sd, full = run_model(ck, seqs)
        results[name] = (ps, pd, sd, full)
        cells = "  ".join(f"g{i+1}:{ps[i]:.2f}" for i in range(L))
        print(f"{name:26s} | {cells:40s} | {full:.2f}")
    print()
    # 진단: 각 모델의 goal1 대비 goal2+ 성능 저하(stitching 갭)
    print("=== stitching 갭 (goal1 → goal2+ 평균 성공률 변화) ===")
    for name, (ps, pd, sd, full) in results.items():
        g1 = ps[0]; g2plus = ps[1:].mean()
        print(f"  {name:26s}  goal1={g1:.2f}  goal2+평균={g2plus:.2f}  저하={g1-g2plus:+.2f}"
              f"   (비-home 시작거리 평균 {np.mean(sd[1:]):.2f}m)")


if __name__ == "__main__":
    main()
