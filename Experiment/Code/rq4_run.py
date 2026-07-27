"""RQ4 프롬프트 요소 ablation — Cartesian 프롬프트에서 섹션 제거 조건별 both·랜덤 배치 실행.

배치:
  (기본)        : ID 정상 씬        → out/cells_rq4v2
  --confusable  : confusable 씬     → out/cells_rq4conf2
  --dist tall   : E2 — 장애물 h 2.0~2.6 m(수직 통과 불가)·좁은 회랑 → 안전 지식이 실제로 binding
                                     → out/cells_rq4tall

조건:
  full            = rq4_full.txt          (전체)
  drop4_safety    = rq4_drop4_safety.txt  (누출 있는 ablation: §4 섹션만 제거, 같은 지식이 §5·§8에 잔존)
  safety_clean    = rq4_safety_clean.txt  (E3 — passable·clearance·경로검사·상승/우회 문구·
                                           출력필드·예시 시연까지 **모든 곳에서** 제거)
  drop2_kinematics, drop8_examples        (기존 v2 조건)

  CUDA_VISIBLE_DEVICES=0 python rq4_run.py --dist tall --conditions full,drop4_safety,safety_clean
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); os.environ.setdefault("MUJOCO_GL", "osmesa")
PROMPTS = os.path.join(HERE, "prompts")
CKPT = os.path.join(ROOT, "Model", "checkpoints", "GCRL_compare", "gcrl_v1_startX_dist.zip")
MODEL = "Qwen/Qwen3.5-27B"
COND = {  # 조건 → 프롬프트 파일 (사용자 재정리 rq4_full.txt 기반)
    "full":             "rq4_full.txt",
    "drop2_kinematics": "rq4_drop2_kinematics.txt",
    "drop4_safety":     "rq4_drop4_safety.txt",
    "drop8_examples":   "rq4_drop8_examples.txt",
    "safety_clean":     "rq4_safety_clean.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=",".join(COND))
    ap.add_argument("--seeds_file", default=os.path.join(HERE, "seeds_mission.json"))
    ap.add_argument("--n_seeds", type=int, default=0, help=">0이면 seeds_file 앞에서 그만큼만(파일럿)")
    ap.add_argument("--dist", default="id", choices=["id", "ood", "tall"])
    ap.add_argument("--sub_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=1800)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--confusable", action="store_true", help="confusable 환경(흙더미·트럭=박스)")
    a = ap.parse_args()
    import mission_run
    seeds = [int(s) for s in json.load(open(a.seeds_file))]
    if a.n_seeds:
        seeds = seeds[:a.n_seeds]
    default_root = ("cells_rq4conf2" if a.confusable else
                    "cells_rq4tall" if a.dist == "tall" else "cells_rq4v2")
    out_root = a.out_root or os.path.join(HERE, "out", default_root)
    for cond in a.conditions.split(","):
        prompt = os.path.join(PROMPTS, COND[cond])
        out_dir = os.path.join(out_root, f"Qwen3.5-27B__both__{a.dist}__{cond}")
        if os.path.exists(os.path.join(out_dir, "results.csv")):
            print(f"[skip] {cond} (results.csv 있음)", flush=True); continue
        print(f"[run] {cond}  prompt={COND[cond]} dist={a.dist} n={len(seeds)} → {out_dir}", flush=True)
        mission_run.run(seeds, MODEL, prompt, "both", a.dist, CKPT, out_dir,
                        a.sub_steps, think=False, refresh=False, gif=False,
                        load_4bit=False, max_steps=a.max_steps, randomize=True, confusable=a.confusable)
    print("[rq4_run] done:", a.conditions, flush=True)


if __name__ == "__main__":
    main()
