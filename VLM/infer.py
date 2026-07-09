"""
VLM 임무 계획 추론 — cabin RGB + 극좌표 3D grid map → 서브골(x,y,z) 시퀀스(JSON).

입력(= make_gridmap.py 산출): VLM/out/cabin_init.png(RGB), VLM/out/gridmap.png(높이맵)
     + system_prompt.txt (+ gridmap_legend.txt 있으면 삽입)
출력: VLM/out/plan.json (scene/objects/subgoals) + goals.json + 콘솔(GCRL용 goal 리스트)
실행: conda run -n seok python VLM/infer.py --instruction "왼쪽 구덩이를 파서 오른쪽 트럭에 실어라"
"""
import os
import re
import json
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))          # .../VLM
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt():
    sp = open(os.path.join(HERE, "system_prompt.txt"), encoding="utf-8").read()
    leg = os.path.join(OUT, "gridmap_legend.txt")           # 색↔높이 정확 범위(프롬프트 삽입용)
    if os.path.exists(leg):
        sp += "\n\n[grid map 색↔높이 legend]\n" + open(leg, encoding="utf-8").read()
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction",
                    default="장면을 분석해 굴착 임무 서브골 시퀀스를 계획하라.")
    ap.add_argument("--rgb", default=os.path.join(OUT, "cabin_init.png"))
    ap.add_argument("--grid", default=os.path.join(OUT, "gridmap.png"))
    ap.add_argument("--max_new_tokens", type=int, default=512)
    a = ap.parse_args()

    for p in (a.rgb, a.grid):
        if not os.path.exists(p):
            raise SystemExit(f"입력 이미지 없음: {p}  (먼저 make_gridmap.py 실행)")

    system_prompt = build_prompt()
    rgb = Image.open(a.rgb).convert("RGB")
    grid = Image.open(a.grid).convert("RGB")

    # ── VLM 로드 (4bit) ──
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── 추론 (system + 두 이미지 + 지시) ──
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": "[이미지 1] RGB cabin view (물체 식별용):"},
            {"type": "image", "image": rgb},
            {"type": "text", "text": "[이미지 2] 3D grid map (위치·높이 기준, top-down 극좌표):"},
            {"type": "image", "image": grid},
            {"type": "text", "text": "지시: " + a.instruction},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[rgb, grid], return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens)
    raw = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("VLM 원본 출력:\n", raw)

    # ── JSON 파싱 ──
    mt = re.search(r"\{.*\}", raw, re.DOTALL)
    if not mt:
        raise SystemExit("JSON을 찾지 못함 — 위 원본 출력 확인.")
    result = json.loads(mt.group())

    # ── 요약 출력 ──
    print("\n=== SCENE ===\n ", result.get("scene", ""))
    print("=== OBJECTS ===")
    for o in result.get("objects", []):
        print(f"  [{o.get('type')}] {o.get('ref')}: "
              f"x={o.get('x')} y={o.get('y')} z={o.get('z')}")
    print("=== SUBGOALS (bucket tip 목표, 순서대로) ===")
    goals = []
    for s in result.get("subgoals", []):
        if all(k in s for k in ("x", "y", "z")):
            x, y, z = float(s["x"]), float(s["y"]), float(s["z"])
            th, r = np.degrees(np.arctan2(x, y)), float(np.hypot(x, y))
            print(f"  {s.get('id')}: (x={x:+.2f}, y={y:+.2f}, z={z:+.2f})  "
                  f"[θ={th:+.0f}° r={r:.2f}m]  ← {s.get('reason', '')}")
            goals.append([round(x, 3), round(y, 3), round(z, 3)])

    # ── 저장 (계획 + GCRL 브리지용 goal 리스트) ──
    json.dump(result, open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"goals": goals}, open(os.path.join(OUT, "goals.json"), "w"), indent=2)
    print(f"\n저장: {OUT}/plan.json, goals.json")
    print("GCRL goals(x,y,z):", goals)
    if goals:
        print("→ 예: python Evaluate/sequential.py --ckpt <run>.zip --goals \""
              + " ".join(f"{g[0]},{g[1]},{g[2]}" for g in goals) + "\"")


if __name__ == "__main__":
    main()
