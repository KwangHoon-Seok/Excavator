"""
A vs B paired 실험 — "VLM 서브골이 obstacle-aware GCRL 성능을 끌어올리는가" 증명.

동일 학습 env(장애물 + 단일 랜덤 goal), 동일 seed로 두 방식을 paired 비교:
  A (baseline) : GCRL이 goal로 **직접** 추종(= FullTaskEval 재현)
  B (VLM-guided): GCRL이 **VLM 서브골 시퀀스**를 relay(각 서브골 도달/timeout 후 다음, 마지막=goal)
좌표계·장애물·goal은 seed로 결정론 재현 → 렌더 씬 == 실행 씬. metric: safe_success/collision/success.
B > A(특히 wide/far goal)이면 VLM 가이드가 회피·도달을 개선한다는 증거.

파이프라인(계획=느림·병렬·캐시 / 실행=빠름·순차):
  1) 렌더  : seed마다 cabin RGB + 장애물 gridmap + scene.json (실행 env 재사용)
  2) 계획  : VLM(claude)로 서브골 — K개 병렬, plan.json 캐시(있으면 skip)
  3) 실행  : A(직접) / B(서브골 relay) 롤아웃 → paired metric

  conda run -n seok python Evaluate/vlm_guided/mission_eval.py --n_seeds 20 --vlm_model haiku --workers 4
VLM 모델 비교: --vlm_model haiku → sonnet → opus (각각 별도 out 서브폴더/결과 CSV).
"""
import os
import sys
import csv
import json
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Model", "Version4"))   # ExcavatorRHEREnv
sys.path.insert(0, os.path.join(ROOT, "VLM"))                 # VLM_infer, gridmap
sys.path.insert(0, HERE)                                      # render_scene
import render_scene                                            # noqa: E402
import VLM_infer                                               # noqa: E402
from stable_baselines3 import SAC                              # noqa: E402

CKPT = os.path.join(ROOT, "Model", "checkpoints", "Version4", "v4base.zip")
PROMPT = "VLM_guidedGCRL_prompt.txt"
INSTRUCTION = "주어진 목표 좌표에 장애물을 피해 도달하는 서브골 시퀀스를 계획하라."


# ── GIF 녹화(정성 분석용) ───────────────────────────────────────
def _sphere(scn, pos, rgba, size=0.15):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0.0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


class Recorder:
    """자유카메라 오프스크린 렌더 → 프레임 누적 → GIF. 현재 목표=초록 큰 구,
    VLM 서브골=파란 작은 구(B에서 경로 시각화). 한 번 생성해 seed마다 reset."""
    def __init__(self, model, h=360, w=480):
        self.r = mujoco.Renderer(model, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0, 1.5, 0.4]
        self.cam.distance = 8.0
        self.cam.azimuth = 135
        self.cam.elevation = -28
        self.frames = []

    def reset(self):
        self.frames = []

    def grab(self, data, target, waypoints=None):
        self.r.update_scene(data, self.cam)
        if waypoints is not None:
            for wp in waypoints:                       # VLM 서브골 경로(작은 파랑)
                _sphere(self.r.scene, wp, [0.2, 0.4, 1.0, 0.7], size=0.10)
        _sphere(self.r.scene, target, [0.1, 0.9, 0.1, 0.85], size=0.18)   # 현재 목표(초록)
        self.frames.append(self.r.render())

    def save(self, path, stride=2, duration=60):
        from PIL import Image
        if not self.frames:
            return
        fr = self.frames[::stride]
        imgs = [Image.fromarray(f) for f in fr]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=duration, loop=0, optimize=True)

    def close(self):
        self.r.close()


# ── 롤아웃 ─────────────────────────────────────────────────────
def rollout_A(env, model, seed, rec=None):
    """baseline — goal 직접 추종. FullTaskEval와 동일 실행/판정."""
    obs, _ = env.reset(seed=seed)
    goal = np.asarray(env._desired, float).copy()
    if rec is not None:
        rec.reset(); rec.grab(env.data, goal)
    collided, info, steps, done = False, {}, 0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        if info.get("min_gap", 1e9) <= 0.0:
            collided = True
        if rec is not None:
            rec.grab(env.data, goal)
        done = term or trunc
    ok = bool(info.get("is_success", False))
    return dict(success=ok, collided=collided, safe=ok and not collided,
                dist=float(info.get("dist", np.nan)), steps=steps)


def rollout_B(env, model, seed, subgoals, sub_steps, rec=None):
    """VLM-guided — 서브골 시퀀스 relay. 마지막에 실제 goal을 붙여 최종 도달 보장.
    각 서브골: eps 이내 도달 or sub_steps 소진 or env 총예산(max_steps) 초과 시 다음으로.
    성공 = 최종적으로 실제 goal에 eps 이내, 충돌 = 전 구간 min_gap≤0 발생."""
    obs, _ = env.reset(seed=seed)
    true_goal = np.asarray(env._desired, float).copy()
    eps = env.eps
    # VLM 서브골 + 실제 goal(마지막). 빈 계획이면 goal 직접(=A와 동일).
    waypoints = [np.asarray(w, float) for w in subgoals] + [true_goal]
    draw_wp = waypoints if rec is not None else None
    collided, total_steps = False, 0
    if rec is not None:
        rec.reset()
    for wp in waypoints:
        env._desired = wp.copy()
        obs = env._get_obs()                      # 새 desired 반영해 관측 갱신
        if rec is not None:
            rec.grab(env.data, wp, waypoints=draw_wp)
        for _ in range(sub_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            total_steps += 1
            if info.get("min_gap", 1e9) <= 0.0:
                collided = True
            if rec is not None:
                rec.grab(env.data, wp, waypoints=draw_wp)
            if float(np.linalg.norm(env._tip() - wp)) < eps:   # 이 서브골 도달 → 다음
                break
            if trunc:                              # env 총예산(max_steps) 소진
                break
        if env._steps >= env.max_steps:
            break
    d_final = float(np.linalg.norm(env._tip() - true_goal))
    ok = d_final < eps
    return dict(success=ok, collided=collided, safe=ok and not collided,
                dist=d_final, steps=total_steps, n_sub=len(subgoals))


# ── goal 구역 분류(FullTaskEval와 동일) ─────────────────────────
def goal_bins(env, goal, start_tip):
    g, s = np.asarray(goal, float), np.asarray(start_tip, float)
    fwd = np.arctan2(s[1], s[0])
    dth = np.arctan2(g[1], g[0]) - fwd
    az_deg = abs(np.degrees(np.arctan2(np.sin(dth), np.cos(dth))))
    d0 = float(np.linalg.norm(g - s))
    azb = "front" if az_deg < 45 else ("side" if az_deg < 90 else "wide")
    db = "near" if d0 < 2.0 else ("mid" if d0 < 3.5 else "far")
    return az_deg, d0, azb, db


# ── VLM 계획(병렬·캐시) ─────────────────────────────────────────
def plan_one(seed, seed_dir, vlm_model, timeout, refresh, prompt=PROMPT):
    """seed 하나 VLM 계획 → goals.json. 캐시 있으면 skip. 실패 시 빈 계획(B는 A로 폴백)."""
    gj = os.path.join(seed_dir, "goals.json")
    if os.path.exists(gj) and not refresh:
        return seed, "cached", json.load(open(gj)).get("goals", [])
    scene = json.load(open(os.path.join(seed_dir, "scene.json")))
    try:
        VLM_infer.infer(INSTRUCTION,
                        os.path.join(seed_dir, "cabin_init.png"),
                        os.path.join(seed_dir, "gridmap.png"),
                        model=vlm_model, timeout=timeout,
                        prompt=prompt, goal=scene["goal"], out=seed_dir)
        return seed, "planned", json.load(open(gj)).get("goals", [])
    except Exception as e:
        with open(os.path.join(seed_dir, "vlm_error.txt"), "w") as f:
            f.write(repr(e) + "\n" + traceback.format_exc())
        return seed, f"FAILED({type(e).__name__})", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20, help="평가 seed 수")
    ap.add_argument("--seed0", type=int, default=300000,
                    help="시작 seed. 기본 300000=학습/eval 미사용 대역(공정한 held-out 분포). "
                         "10000/50000/900000 대역은 학습중 eval에 쓰였으니 회피")
    ap.add_argument("--vlm_model", default="haiku", help="claude 모델: haiku|sonnet|opus (탐색 순서)")
    ap.add_argument("--workers", type=int, default=4, help="VLM 병렬 호출 수")
    ap.add_argument("--timeout", type=int, default=1200, help="VLM 호출당 타임아웃(초)")
    ap.add_argument("--sub_steps", type=int, default=150, help="B: 서브골당 최대 스텝(내부 가드)")
    ap.add_argument("--max_steps_B", type=int, default=None, help="B 총 스텝예산(기본=config max_steps, A와 동일)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seeds_file", default=None,
                    help="유효 seed 리스트 json(예: valid_seeds.json). 주면 seed0/n_seeds 대신 사용")
    ap.add_argument("--prompt", default=PROMPT,
                    help="VLM system prompt 파일(기본=기본 reach 프롬프트 / smooth=VLM_guidedGCRL_smooth_prompt.txt)")
    ap.add_argument("--refresh", action="store_true", help="VLM 계획 캐시 무시하고 재호출")
    ap.add_argument("--no_gif", action="store_true", help="롤아웃 GIF 저장 끄기(기본 저장)")
    a = ap.parse_args()

    if a.seeds_file:
        seeds = [int(s) for s in json.load(open(a.seeds_file))]
    else:
        seeds = list(range(a.seed0, a.seed0 + a.n_seeds))
    run_root = os.path.join(a.out, a.vlm_model)               # 모델별 결과 분리
    os.makedirs(run_root, exist_ok=True)

    env = render_scene.build_env()
    env.set_curriculum(1.0)                                   # 전체 task(FullTaskEval와 동일 분포)
    model = SAC.load(a.ckpt, env=env, device="auto")         # HerReplayBuffer라 env 필수(predict만 쓰지만)
    print(f"ckpt={a.ckpt}\nseeds={seeds[0]}..{seeds[-1]} ({a.n_seeds})  vlm={a.vlm_model}  workers={a.workers}")

    # ── 1) 렌더(순차, 실행 env 재사용 → 렌더 씬 == 실행 씬) ──
    print("\n[1/3] 씬 렌더...")
    scenes = {}
    for s in seeds:
        env.reset(seed=s)
        d = os.path.join(run_root, f"seed{s}")
        scenes[s] = render_scene.render_current(env, d, s)
    print(f"  {len(scenes)}개 씬 렌더 완료 → {run_root}/seed*/")

    # ── 2) VLM 계획(병렬·캐시) ──
    print(f"\n[2/3] VLM 계획(model={a.vlm_model}, 병렬 {a.workers})... 오래 걸릴 수 있음")
    plans = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(plan_one, s, os.path.join(run_root, f"seed{s}"),
                          a.vlm_model, a.timeout, a.refresh, a.prompt) for s in seeds]
        for f in futs:
            s, status, goals = f.result()
            plans[s] = goals
            print(f"  seed {s}: {status}  subgoals={len(goals)}")

    # ── 3) 실행(A/B 순차, paired) ──
    print("\n[3/3] A/B 롤아웃%s..." % ("" if a.no_gif else " + GIF 녹화"))
    cfg_max = env.max_steps                                   # config 총 스텝예산(A용)
    max_steps_B = a.max_steps_B or cfg_max                    # B 총예산(기본=A와 동일 → 공정 비교)
    rec = None if a.no_gif else Recorder(env.model)
    rows = []
    for s in seeds:
        seed_dir = os.path.join(run_root, f"seed{s}")
        env.max_steps = cfg_max
        rA = rollout_A(env, model, s, rec=rec)
        if rec is not None:
            rec.save(os.path.join(seed_dir, "rollout_A.gif"))
        env.max_steps = max_steps_B
        rB = rollout_B(env, model, s, plans.get(s, []), a.sub_steps, rec=rec)
        if rec is not None:
            rec.save(os.path.join(seed_dir, "rollout_B.gif"))
        az_deg, d0, azb, db = goal_bins(env, scenes[s]["goal"], scenes[s]["start_tip"])
        rows.append(dict(seed=s, az_deg=round(az_deg, 1), d0=round(d0, 2), azb=azb, db=db,
                         n_obs=len(scenes[s]["obstacles"]), n_sub=rB["n_sub"],
                         A_succ=int(rA["success"]), A_coll=int(rA["collided"]), A_safe=int(rA["safe"]),
                         A_dist=round(rA["dist"], 3),
                         B_succ=int(rB["success"]), B_coll=int(rB["collided"]), B_safe=int(rB["safe"]),
                         B_dist=round(rB["dist"], 3)))
        print(f"  seed {s} [{azb}/{db} az={az_deg:.0f}° d0={d0:.1f} sub={rB['n_sub']}] "
              f"A(safe={rA['safe']:d} succ={rA['success']:d} coll={rA['collided']:d}) "
              f"B(safe={rB['safe']:d} succ={rB['success']:d} coll={rB['collided']:d})")

    # ── 집계 + 저장 ──
    csv_path = os.path.join(run_root, f"results_{a.vlm_model}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def rate(key):
        return np.mean([r[key] for r in rows])
    n = len(rows)
    print("\n" + "=" * 64)
    print(f"결과 ({n} seeds, vlm={a.vlm_model})   [paired A=직접 vs B=VLM서브골]")
    print("-" * 64)
    print(f"{'metric':<16}{'A (baseline)':>16}{'B (VLM)':>14}{'Δ (B−A)':>14}")
    for key, name in [("safe", "safe_success"), ("succ", "success"), ("coll", "collision")]:
        a_r, b_r = rate(f"A_{key}"), rate(f"B_{key}")
        print(f"{name:<16}{a_r:>16.3f}{b_r:>14.3f}{b_r - a_r:>+14.3f}")
    print("-" * 64)
    print("구역별 safe_success (A → B):")
    for grp in ("front", "side", "wide"):
        sub = [r for r in rows if r["azb"] == grp]
        if sub:
            aa = np.mean([r["A_safe"] for r in sub]); bb = np.mean([r["B_safe"] for r in sub])
            print(f"  {grp:<6} (n={len(sub):2d}):  A={aa:.2f} → B={bb:.2f}  ({bb-aa:+.2f})")
    for grp in ("near", "mid", "far"):
        sub = [r for r in rows if r["db"] == grp]
        if sub:
            aa = np.mean([r["A_safe"] for r in sub]); bb = np.mean([r["B_safe"] for r in sub])
            print(f"  {grp:<6} (n={len(sub):2d}):  A={aa:.2f} → B={bb:.2f}  ({bb-aa:+.2f})")
    print("=" * 64)
    print("saved 정량:", csv_path)
    if rec is not None:
        print("saved 정성: %s/seed*/rollout_{A,B}.gif" % run_root)
        rec.close()
    env.close()


if __name__ == "__main__":
    main()
