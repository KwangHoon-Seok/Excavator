"""
Qwen-VLM-guided GCRL A/B 실험 러너.

동일 학습 env(장애물 + 단일 랜덤 goal), 동일 seed로 두 방식을 paired 비교:
  A (baseline)  : GCRL이 goal로 **직접** 추종
  B (VLM-guided): **로컬 Qwen2.5-VL** 서브골 시퀀스를 relay(각 서브골 도달/timeout 후 다음, 마지막=goal)

VLM/VLM_infer.py(Claude CLI) 대신 Experiment/Code/qwen_planner.py(로컬 Qwen)를 planner로 쓰는 것만 다르고,
씬 렌더·롤아웃·metric·GIF는 Evaluate/vlm_guided/{render_scene,mission_eval}를 그대로 재사용한다.
Qwen 출력(plan.json)은 VLM/interface.py:parse_plan을 통해 GCRL goal 시퀀스로 변환한다.

파이프라인:
  1) 렌더  : seed마다 cabin RGB + 장애물 gridmap + scene.json (실행 env 재사용 → 렌더 씬 == 실행 씬)
  2) 계획  : Qwen(로컬, GPU) 서브골 — 순차(모델 싱글턴), plan.json 캐시(있으면 skip)
  3) 실행  : A(직접) / B(서브골 relay) 롤아웃 → paired metric CSV + GIF

실행:
  python Experiment/Code/run_experiment.py --n_seeds 2 --no_gif                 # 스모크
  python Experiment/Code/run_experiment.py --seeds_file Evaluate/vlm_guided/valid_seeds.json
  python Experiment/Code/run_experiment.py --qwen_model Qwen/Qwen2.5-VL-32B-Instruct   # 더 큰 모델
"""
import os
import sys
import csv
import json
import argparse
import traceback

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")                     # headless 소프트웨어 렌더(디스플레이 없음)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                     # repo 루트
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))  # render_scene, mission_eval
sys.path.insert(0, os.path.join(ROOT, "VLM"))                     # interface
sys.path.insert(0, HERE)                                          # qwen_planner

import render_scene                                               # noqa: E402
import mission_eval as ME                                         # noqa: E402  (rollout/Recorder/goal_bins 재사용)
from interface import parse_plan                                  # noqa: E402  (Qwen plan → GCRL goals)
from stable_baselines3 import SAC                                 # noqa: E402
import qwen_planner                                               # noqa: E402

CKPT = os.path.join(ROOT, "Model", "checkpoints", "Version4", "v4base.zip")
PROMPT = "VLM_guidedGCRL_prompt.txt"
INSTRUCTION = ME.INSTRUCTION      # mission_eval와 동일 지시문(공정 비교)


def plan_one(seed_dir, goal, model_id, prompt, refresh):
    """seed 하나 Qwen 계획 → plan.json → interface.parse_plan → goals.
    캐시(plan.json) 있으면 skip. 실패 시 빈 계획(B는 A로 폴백)."""
    pj = os.path.join(seed_dir, "plan.json")
    if os.path.exists(pj) and not refresh:
        return parse_plan(pj)["goals"], "cached"
    try:
        qwen_planner.infer(INSTRUCTION,
                           os.path.join(seed_dir, "cabin_init.png"),
                           os.path.join(seed_dir, "gridmap.png"),
                           model=model_id, prompt=prompt, goal=goal, out=seed_dir)
        return parse_plan(pj)["goals"], "planned"
    except Exception as e:
        with open(os.path.join(seed_dir, "vlm_error.txt"), "w") as f:
            f.write(repr(e) + "\n" + traceback.format_exc())
        return [], f"FAILED({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20, help="평가 seed 수")
    ap.add_argument("--seed0", type=int, default=300000,
                    help="시작 seed. 기본 300000=학습/eval 미사용 held-out 대역")
    ap.add_argument("--qwen_model", default=qwen_planner.MODEL_ID,
                    help="HF 모델 id(예: Qwen/Qwen2.5-VL-7B-Instruct | ...-32B-Instruct | ...-72B-Instruct)")
    ap.add_argument("--sub_steps", type=int, default=150, help="B: 서브골당 최대 스텝(내부 가드)")
    ap.add_argument("--max_steps_B", type=int, default=None, help="B 총 스텝예산(기본=config max_steps, A와 동일)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seeds_file", default=None,
                    help="유효 seed 리스트 json(예: Evaluate/vlm_guided/valid_seeds.json). 주면 seed0/n_seeds 대신")
    ap.add_argument("--prompt", default=PROMPT,
                    help="VLM system prompt 파일(기본 reach / smooth=VLM_guidedGCRL_smooth_prompt.txt)")
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="Qwen 생성 토큰 상한")
    ap.add_argument("--refresh", action="store_true", help="Qwen 계획 캐시 무시하고 재호출")
    ap.add_argument("--no_gif", action="store_true", help="롤아웃 GIF 저장 끄기(기본 저장)")
    a = ap.parse_args()

    if a.seeds_file:
        seeds = [int(s) for s in json.load(open(a.seeds_file))]
    else:
        seeds = list(range(a.seed0, a.seed0 + a.n_seeds))
    tag = a.qwen_model.split("/")[-1]                             # 모델별 결과 분리
    run_root = os.path.join(a.out, tag)
    os.makedirs(run_root, exist_ok=True)

    env = render_scene.build_env()
    env.set_curriculum(1.0)                                       # 전체 task 분포(FullTaskEval와 동일)
    model = SAC.load(a.ckpt, env=env, device="auto")             # HerReplayBuffer라 env 필수(predict만 쓰지만)
    print(f"ckpt={a.ckpt}\nseeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  qwen={a.qwen_model}")

    # ── 1) 렌더(순차, 실행 env 재사용 → 렌더 씬 == 실행 씬) ──
    print("\n[1/3] 씬 렌더...")
    scenes = {}
    for s in seeds:
        env.reset(seed=s)
        scenes[s] = render_scene.render_current(env, os.path.join(run_root, f"seed{s}"), s)
    print(f"  {len(scenes)}개 씬 렌더 완료 → {run_root}/seed*/")

    # ── 2) Qwen 계획(순차: GPU 싱글턴 · plan.json 캐시) ──
    print(f"\n[2/3] Qwen 계획(model={a.qwen_model}, 순차)... 첫 seed에서 모델 로드(가중치 다운로드 가능)")
    plans = {}
    for s in seeds:
        seed_dir = os.path.join(run_root, f"seed{s}")
        goals, status = plan_one(seed_dir, scenes[s]["goal"], a.qwen_model, a.prompt, a.refresh)
        plans[s] = goals
        print(f"  seed {s}: {status}  subgoals={len(goals)}")

    # ── 3) 실행(A/B 순차, paired) ──
    print("\n[3/3] A/B 롤아웃%s..." % ("" if a.no_gif else " + GIF 녹화"))
    cfg_max = env.max_steps                                       # config 총 스텝예산(A용)
    max_steps_B = a.max_steps_B or cfg_max                        # B 총예산(기본=A와 동일 → 공정 비교)
    rec = None if a.no_gif else ME.Recorder(env.model)
    rows = []
    for s in seeds:
        seed_dir = os.path.join(run_root, f"seed{s}")
        env.max_steps = cfg_max
        rA = ME.rollout_A(env, model, s, rec=rec)
        if rec is not None:
            rec.save(os.path.join(seed_dir, "rollout_A.gif"))
        env.max_steps = max_steps_B
        rB = ME.rollout_B(env, model, s, plans.get(s, []), a.sub_steps, rec=rec)
        if rec is not None:
            rec.save(os.path.join(seed_dir, "rollout_B.gif"))
        az_deg, d0, azb, db = ME.goal_bins(env, scenes[s]["goal"], scenes[s]["start_tip"])
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
    csv_path = os.path.join(run_root, f"results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def rate(key):
        return float(np.mean([r[key] for r in rows]))
    n = len(rows)
    print("\n" + "=" * 64)
    print(f"결과 ({n} seeds, qwen={tag})   [paired A=직접 vs B=Qwen서브골]")
    print("-" * 64)
    print(f"{'metric':<16}{'A (baseline)':>16}{'B (Qwen)':>14}{'Δ (B−A)':>14}")
    for key, name in [("safe", "safe_success"), ("succ", "success"), ("coll", "collision")]:
        a_r, b_r = rate(f"A_{key}"), rate(f"B_{key}")
        print(f"{name:<16}{a_r:>16.3f}{b_r:>14.3f}{b_r - a_r:>+14.3f}")
    print("-" * 64)
    print("구역별 safe_success (A → B):")
    for grp in ("front", "side", "wide", "near", "mid", "far"):
        sub = [r for r in rows if r["azb"] == grp or r["db"] == grp]
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
