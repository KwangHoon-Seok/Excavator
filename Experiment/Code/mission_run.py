"""
상차 미션 A/B 실험 러너 — 흙더미→트럭(장애물 회피).
  A (baseline)  : GCRL이 [PILE_PICK, TRUCK_DUMP] 두 목표로 **직접**(중간 회피 waypoint 없음)
  B (VLM-guided): 로컬 Qwen이 계획한 서브골 시퀀스를 relay

파라미터: --qwen_model / --prompt / --modality / --dist(id|ood). 결과 CSV + (옵션)GIF.
RQ 오케스트레이션은 run_all_rqs.py가 이 파일을 조합해 호출.

  python Experiment/Code/mission_run.py --qwen_model Qwen/Qwen3.5-4B --n_seeds 2 --no_gif
"""
import os
import sys
import csv
import json
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))  # mission_eval(Recorder)
sys.path.insert(0, os.path.join(ROOT, "VLM"))                   # interface
sys.path.insert(0, HERE)

import mission as M                                             # noqa: E402
import mission_eval as ME                                       # noqa: E402  (Recorder 재사용)
from interface import parse_plan                                # noqa: E402
from stable_baselines3 import SAC                               # noqa: E402
import qwen_planner                                             # noqa: E402

CKPT = os.path.join(ROOT, "Model", "checkpoints", "Version4", "v4base.zip")
DEFAULT_PROMPT = os.path.join(ROOT, "VLM", "system_prompt.txt")  # 상차(loading) 프롬프트


def plan_one(seed_dir, scene, model_id, prompt, modality, think, refresh, load_4bit=False):
    """seed 하나 Qwen 계획 → plan.json → interface.parse_plan → goals. 캐시/폴백."""
    pj = os.path.join(seed_dir, "plan.json")
    if os.path.exists(pj) and not refresh:
        return parse_plan(pj)["goals"], "cached"
    try:
        qwen_planner.infer(M.INSTRUCTION,
                           os.path.join(seed_dir, "cabin_init.png"),
                           os.path.join(seed_dir, "gridmap.png"),
                           model=model_id, prompt=prompt, goal=None, out=seed_dir,
                           modality=modality, think=think, load_4bit=load_4bit)
        return parse_plan(pj)["goals"], "planned"
    except Exception as e:
        with open(os.path.join(seed_dir, "vlm_error.txt"), "w") as f:
            f.write(repr(e) + "\n" + traceback.format_exc())
        return [], f"FAILED({type(e).__name__})"


def run(seeds, qwen_model, prompt, modality, dist, ckpt, out_root, sub_steps,
        think=False, refresh=False, gif=False, load_4bit=False, max_steps=1000, randomize=False,
        confusable=False):
    """A/B 미션 실험 한 세트 실행 → results.csv + rows 반환. (RQ 러너가 조건별 호출)
    max_steps=미션 총 스텝예산(A/B 동일, pile→truck 2연속 선회라 config 400보다 크게)."""
    os.makedirs(out_root, exist_ok=True)
    env = M.build_gcrl_env()
    env.set_curriculum(1.0)
    env.max_steps = max_steps                                   # 미션 총예산(A/B 공통)
    model = SAC.load(ckpt, env=env, device="auto")
    print(f"[run] model={qwen_model} prompt={os.path.basename(prompt)} modality={modality} "
          f"dist={dist} seeds={len(seeds)} → {out_root}")

    # 1) 렌더(미션 씬: 흙더미+트럭+장애물)
    scenes = {}
    for s in seeds:
        scenes[s] = M.render_mission(s, os.path.join(out_root, f"seed{s}"), dist=dist,
                                     randomize=randomize, confusable=confusable)

    # 2) Qwen 계획(순차, GPU 싱글턴)
    plans = {}
    for s in seeds:
        goals, status = plan_one(os.path.join(out_root, f"seed{s}"), scenes[s],
                                 qwen_model, prompt, modality, think, refresh, load_4bit)
        plans[s] = goals
        print(f"  seed {s}: {status}  subgoals={len(goals)}")

    # 3) A/B 롤아웃
    rec = ME.Recorder(env.model) if gif else None
    rows = []
    for s in seeds:
        obs_boxes = scenes[s]["obstacles"]
        tgt = dict(pile_pick=scenes[s].get("pile_pick"), truck_dump=scenes[s].get("truck_dump"),
                   truck_fp=scenes[s].get("truck_fp"))                       # per-seed 타깃(랜덤화)
        rA = M.rollout_mission(env, model, s, obs_boxes, [], sub_steps=sub_steps, rec=rec, **tgt)  # A: [pile,truck]
        if rec is not None:
            rec.save(os.path.join(out_root, f"seed{s}", "rollout_A.gif"))
        rB = M.rollout_mission(env, model, s, obs_boxes, plans.get(s, []), sub_steps=sub_steps, rec=rec, **tgt)
        if rec is not None:
            rec.save(os.path.join(out_root, f"seed{s}", "rollout_B.gif"))
        rows.append(dict(seed=s, dist=dist, modality=modality, model=qwen_model.split("/")[-1],
                         prompt=os.path.basename(prompt), n_obs=len(obs_boxes), n_sub=rB["n_sub"],
                         A_success=rA["success"], A_coll=rA["collided"], A_safe=rA["safe"],
                         A_stage1=rA["stage1"], A_stage2=rA["stage2"], A_path=rA["path"],
                         B_success=rB["success"], B_coll=rB["collided"], B_safe=rB["safe"],
                         B_stage1=rB["stage1"], B_stage2=rB["stage2"], B_path=rB["path"]))
        print(f"  seed {s} [{dist} obs={len(obs_boxes)} sub={rB['n_sub']}] "
              f"A(safe={rA['safe']} s1={rA['stage1']} s2={rA['stage2']} coll={rA['collided']}) "
              f"B(safe={rB['safe']} s1={rB['stage1']} s2={rB['stage2']} coll={rB['collided']})")
    if rec is not None:
        rec.close()
    env.close()

    csv_path = os.path.join(out_root, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[run] saved {csv_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="상차 system prompt(기본 VLM/system_prompt.txt)")
    ap.add_argument("--modality", default="both", choices=qwen_planner.MODALITIES)
    ap.add_argument("--dist", default="id", choices=["id", "ood"])
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--seeds_file", default=None, help="seed 리스트 json(주면 n_seeds 무시)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=None, help="출력 폴더(기본 out/mission/<model>)")
    ap.add_argument("--sub_steps", type=int, default=200, help="서브골당 최대 스텝(내부 가드)")
    ap.add_argument("--max_steps", type=int, default=1000, help="미션 총 스텝예산(A/B 동일)")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no_gif", action="store_true")
    a = ap.parse_args()
    if a.seeds_file:
        seeds = [int(s) for s in json.load(open(a.seeds_file))]
    else:
        seeds = list(range(a.seed0, a.seed0 + a.n_seeds))
    out = a.out or os.path.join(HERE, "out", "mission", a.qwen_model.split("/")[-1])
    run(seeds, a.qwen_model, a.prompt, a.modality, a.dist, a.ckpt, out, a.sub_steps,
        think=a.think, refresh=a.refresh, gif=not a.no_gif, load_4bit=a.load_4bit,
        max_steps=a.max_steps)


if __name__ == "__main__":
    main()
