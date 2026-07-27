"""
RQ 전체 오케스트레이터 — 상차 미션 A/B를 조건별(cell)로 실행.
셀 = (model, modality, dist, prompt). RQ 간 공유 셀(27B/both/id/full)은 한 번만 실행(dedup).
결과는 out/cells/<cellid>/results.csv. 재개 가능(있으면 skip). 모델 그룹 단위로 GPU/디스크 정리.

RQ 매핑:
  RQ1 모달리티 : 27B × {both, rgb_only, height_only} (id, full)
  RQ2 크기     : {4B, 27B, 35B-A3B, [122B]} (both, id, full)
  RQ3 OOD      : 27B × {id, ood} (both, full)  — A(단독GCRL) vs B(VLM+GCRL) 대조
  RQ4 프롬프트 : 27B × {full, no_kinematics, no_cot, no_clearance, minimal} (both, id)

  python Experiment/Code/run_all_rqs.py                 # 전체
  python Experiment/Code/run_all_rqs.py --only 4B       # 특정 모델 그룹만
"""
import os
import sys
import json
import gc
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "osmesa")

PROMPTS = os.path.join(HERE, "prompts")
FIXED = "Qwen/Qwen3.5-35B-A3B"                                   # RQ3/4 고정 모델(MoE 3B활성 → 빠름·고품질)
RQ1_MODEL = "Qwen/Qwen3.5-27B"                                   # RQ1 모달리티 ablation 고정 모델(크기 격리, dense 27B)
LADDER = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B"]  # RQ2 (+122B 조건부)
P_FULL = os.path.join(PROMPTS, "rq4_full.txt")


def _cell_id(model, modality, dist, prompt):
    return f"{model.split('/')[-1]}__{modality}__{dist}__{os.path.basename(prompt).replace('.txt','')}"


def build_cells(include_122b=None):
    """dedup된 셀 목록 + RQ→cellid 매핑. include_122b=양자화 122B repo id(없으면 None)."""
    cells = {}   # id -> dict

    def add(model, modality, dist, prompt, load_4bit=False):
        cid = _cell_id(model, modality, dist, prompt)
        cells[cid] = dict(id=cid, model=model, modality=modality, dist=dist,
                          prompt=prompt, load_4bit=load_4bit)
        return cid

    C0 = add(FIXED, "both", "id", P_FULL)                       # 공유 셀(RQ1 both = RQ2 27B = RQ3 id = RQ4 full)
    rq = {"RQ1": {}, "RQ2": {}, "RQ3": {}, "RQ4": {}}

    # RQ1 모달리티 — 크기 격리를 위해 27B 고정(README 규약). both는 RQ2의 27B 셀과 동일 id → dedup 재사용.
    rq["RQ1"]["both"] = add(RQ1_MODEL, "both", "id", P_FULL)
    rq["RQ1"]["rgb_only"] = add(RQ1_MODEL, "rgb_only", "id", os.path.join(PROMPTS, "rq1_rgb_only.txt"))
    rq["RQ1"]["height_only"] = add(RQ1_MODEL, "height_only", "id", os.path.join(PROMPTS, "rq1_height_only.txt"))

    # RQ2 크기 사다리
    for m in LADDER:
        rq["RQ2"][m.split("/")[-1]] = add(m, "both", "id", P_FULL)
    if include_122b:
        rq["RQ2"]["Qwen3.5-122B-A10B"] = add(include_122b, "both", "id", P_FULL, load_4bit=True)

    # RQ3 OOD
    rq["RQ3"]["id"] = C0
    rq["RQ3"]["ood"] = add(FIXED, "both", "ood", P_FULL)

    # RQ4 프롬프트
    rq["RQ4"]["full"] = C0
    for v in ("no_kinematics", "no_cot", "no_clearance", "minimal"):
        rq["RQ4"][v] = add(FIXED, "both", "id", os.path.join(PROMPTS, f"rq4_{v}.txt"))

    return cells, rq


def _evict_model(model):
    """HF 허브 캐시에서 모델 삭제(디스크 확보)."""
    name = "models--" + model.replace("/", "--")
    p = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), name)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print(f"[evict] {p}")


def _free_gpu():
    import qwen_planner
    import torch
    qwen_planner._CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_file", default=os.path.join(HERE, "seeds_mission.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cells"))
    ap.add_argument("--sub_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=1800)   # 다중 서브골(B) 여유 예산(A는 조기 종료)
    ap.add_argument("--only", default=None, help="특정 모델 tag만(예: 4B) 실행")
    ap.add_argument("--shard", default=None, help="셀 분할 'M/N'(병렬 워커용, 그룹 내 index%%N==M만). CUDA_VISIBLE_DEVICES와 함께 GPU별 병렬")
    ap.add_argument("--no_evict", action="store_true", help="모델 그룹 후 HF 캐시 삭제 안 함")
    ap.add_argument("--include_122b", default=None, help="양자화 122B repo id(있으면 RQ2에 추가)")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    import mission_run
    seeds = [int(s) for s in json.load(open(a.seeds_file))]
    cells, rq = build_cells(a.include_122b)
    json.dump(rq, open(os.path.join(HERE, "rq_cell_map.json"), "w"), indent=2)
    print(f"seeds={len(seeds)}  cells={len(cells)}  → {a.out}")

    # 모델 그룹 순서(27B 먼저: 셀 다수 / 그다음 4B / 35B / 122B). 그룹 종료 시 evict.
    order = [FIXED, "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B"] + (
        [a.include_122b] if a.include_122b else [])
    by_model = {}
    for c in cells.values():
        by_model.setdefault(c["model"], []).append(c)

    shard_m = shard_n = None
    if a.shard:
        shard_m, shard_n = (int(x) for x in a.shard.split("/"))

    for model in order:
        group = by_model.get(model, [])
        if not group or (a.only and a.only not in model):
            continue
        group = sorted(group, key=lambda c: c["id"])
        if shard_n:
            group = [c for i, c in enumerate(group) if i % shard_n == shard_m]
        if not group:
            continue
        print(f"\n===== 모델 그룹: {model}  ({len(group)} cells, shard={a.shard}) =====")
        for c in group:
            out_root = os.path.join(a.out, c["id"])
            if os.path.exists(os.path.join(out_root, "results.csv")) and not a.refresh:
                print(f"[skip] {c['id']} (results.csv 있음)")
                continue
            mission_run.run(seeds, c["model"], c["prompt"], c["modality"], c["dist"],
                            mission_run.CKPT, out_root, a.sub_steps,
                            think=False, refresh=a.refresh, gif=False,
                            load_4bit=c["load_4bit"], max_steps=a.max_steps)
        _free_gpu()
        if not a.no_evict and not (a.only):
            _evict_model(model)
    print("\n[run_all_rqs] 완료. aggregate_rqs.py로 figure/table 생성.")


if __name__ == "__main__":
    main()
