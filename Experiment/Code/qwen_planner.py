"""
Qwen VLM 임무계획 추론 — 로컬 HuggingFace VLM으로 RGB + gridmap → 서브골 시퀀스 JSON.

VLM/VLM_infer.py(Claude CLI 판)의 드롭인 대체: 같은 입력(시스템 프롬프트 + 두 이미지 + 지시(+goal))을
받아 같은 출력(plan.json / goals.json)을 낸다. 차이는 VLM이 **로컬 Qwen(transformers)**라는 점.

모델-무관 로더(`AutoModelForImageTextToText`)라 Qwen2.5-VL과 **Qwen3.5(네이티브 멀티모달)** 모두 로드된다.
Qwen3.5는 thinking 모드가 기본이라 출력 앞에 <think>…</think> 추론이 붙을 수 있어 JSON 추출을
thinking-robust하게 처리한다(</think> 뒤 + 마지막 balanced {…}).

RQ1(멀티모달 ablation)용 `modality`:
  both        : RGB + gridmap (기본)
  rgb_only    : RGB만
  height_only : gridmap만

입력: RGB(cabin_init.png) + gridmap.png + system prompt(--prompt) (+ 같은 폴더 gridmap_legend.txt)
출력: plan.json(scene/objects/subgoals), goals.json(GCRL용 goal 리스트) — --out 폴더에 저장

실행(단일 씬 테스트):
  python Experiment/Code/qwen_planner.py --model Qwen/Qwen3.5-4B \
      --rgb  Experiment/Code/out/.../cabin_init.png \
      --grid Experiment/Code/out/.../gridmap.png \
      --out  Experiment/Code/out/.../ --goal 1.2,3.4,0.3
"""
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # Experiment/Code → repo 루트
VLM_DIR = os.path.join(ROOT, "VLM")                    # 기본 system prompt 파일 위치
PROMPTS_DIR = os.path.join(HERE, "prompts")            # RQ1/RQ4 프롬프트 변형
OUT = os.path.join(HERE, "out")

MODEL_ID = "Qwen/Qwen3.5-4B"                           # 사다리 시작(4B). --model로 교체
MODALITIES = ("both", "rgb_only", "height_only")

# ── 모델 싱글턴((id,dtype,4bit)별 캐시 — 로드가 느리므로 재사용) ──
_CACHE = {}


def _torch_dtype(name):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name, torch.bfloat16)


def load_model(model_id=MODEL_ID, dtype="bf16", load_4bit=False):
    """AutoModelForImageTextToText 로드(+캐시). 큰 모델(122B 등)은 load_4bit=True로 2×A100에 적재."""
    key = (model_id, dtype, load_4bit)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kwargs = dict(device_map="auto")
    try:                                               # 하이브리드 full-attn 층 가속(긴 프롬프트 decode)
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        pass
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        # 4bit는 CPU/disk 오프로드 불가 → 보이는 단일 GPU에 강제 배치(auto가 잘못 spill하는 것 방지).
        # 다중 GPU가 필요한 초대형(122B)만 QWEN_4BIT_AUTO=1로 auto 유지.
        if os.environ.get("QWEN_4BIT_AUTO") != "1":
            kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = _torch_dtype(dtype)          # transformers 5.x 표준(구버전은 torch_dtype)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except TypeError:                                  # 구버전 transformers: dtype= 미지원 → torch_dtype=
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    _CACHE[key] = (model, processor)
    return model, processor


def _resolve_prompt(prompt):
    """프롬프트 파일 경로 해석 — 절대경로면 그대로, 아니면 Experiment/Code/prompts/ → VLM/ 순 탐색."""
    if os.path.isabs(prompt) or os.path.exists(prompt):
        return prompt
    for base in (PROMPTS_DIR, VLM_DIR):
        p = os.path.join(base, prompt)
        if os.path.exists(p):
            return p
    return os.path.join(VLM_DIR, prompt)               # 폴백(에러 메시지용)


def build_messages(instruction, rgb, grid, prompt="VLM_guidedGCRL_prompt.txt",
                   goal=None, legend=None, modality="both", brief=False):
    """Qwen chat 메시지 구성: system(프롬프트+legend) + user[포함 이미지들 + 지시(+goal)].
    modality로 RGB/heightmap 포함 여부 결정(RQ1). legend는 gridmap 있을 때만 append."""
    assert modality in MODALITIES, f"modality는 {MODALITIES} 중 하나"
    use_rgb = modality in ("both", "rgb_only")
    use_grid = modality in ("both", "height_only")

    sp = open(_resolve_prompt(prompt), encoding="utf-8").read()
    if legend is None:                                 # grid 옆 gridmap_legend.txt 자동 탐색
        legend = os.path.join(os.path.dirname(os.path.abspath(grid)), "gridmap_legend.txt")
    if use_grid and legend and os.path.exists(legend):
        sp += "\n\n[grid map 색↔높이 legend]\n" + open(legend, encoding="utf-8").read()

    goal_line = ""
    if goal is not None:
        goal_line = (f"목표 좌표(버킷 끝 도달점): ({float(goal[0]):.3f}, {float(goal[1]):.3f}, "
                     f"{float(goal[2]):.3f}) [m]. 이 좌표가 최종 서브골이며, 장애물을 피해 도달하라.\n")
    user_text = (f"{goal_line}지시: {instruction}\n\n"
                 f"위 이미지를 보고, 섹션 10 형식의 JSON만 출력하라. JSON 밖의 텍스트/마크다운 금지.")
    if brief:                                          # 생성 토큰 절감(좌표 품질엔 영향 최소, 설명필드만 생략)
        user_text += ("\n\n[출력 간결화] 내부적으로는 모든 규칙(운동학·passable·clearance·경로검사)을 "
                      "충실히 따르되, **출력 JSON에는 설명 문자열을 넣지 마라**: subgoals는 {id,x,y,z}만, "
                      "objects는 {ref,type,x,y,z,h_max,passable}만. reason/clearance_check/passable_reason/scene "
                      "필드는 생략. 수치는 소수 둘째 자리.")

    from PIL import Image
    content = []
    if use_rgb:
        content.append({"type": "text", "text": "[이미지 1] RGB cabin view (장애물 식별용):"})
        content.append({"type": "image", "image": Image.open(rgb).convert("RGB")})
    if use_grid:
        content.append({"type": "text", "text": "[이미지 2] 3D grid map (장애물 위치·높이, top-down 극좌표):"})
        content.append({"type": "image", "image": Image.open(grid).convert("RGB")})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": sp}]},
        {"role": "user", "content": content},
    ]


def generate(messages, model_id=MODEL_ID, dtype="bf16", load_4bit=False,
             max_new_tokens=1536, think=False):
    """메시지 → Qwen 생성 → 순수 텍스트(생성분만) 반환. greedy(재현성).
    think=True면 Qwen3.5 thinking 모드(추론 후 JSON, 토큰 많이 필요). 기본 off(대량 실험 속도)."""
    import torch
    from qwen_vl_utils import process_vision_info
    model, processor = load_model(model_id, dtype, load_4bit)
    try:
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=think)
    except TypeError:                                  # 템플릿이 enable_thinking 미지원(구 모델)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def _extract_json(raw):
    """thinking-robust JSON 추출: </think> 뒤만 보고, 마지막 balanced {…}를 파싱.
    (Qwen3.5 추론 텍스트에 중괄호가 섞여도 최종 JSON을 안정적으로 잡는다.)"""
    body = raw.rsplit("</think>", 1)[1] if "</think>" in raw else raw
    end = body.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if body[i] == "}":
                depth += 1
            elif body[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[i:end + 1])
                    except Exception:
                        break                          # 이 후보 실패 → 더 앞의 } 로
        end = body.rfind("}", 0, end)
    raise ValueError("Qwen 출력에서 JSON 파싱 실패 — 원본:\n" + raw[:1000])


def parse_and_save(raw, out=OUT):
    """모델 텍스트 → plan.json, goals.json 저장. (plan, goals) 반환.
    JSON 실패 시 ValueError(배치 러너의 except Exception이 잡아 폴백)."""
    plan = _extract_json(raw)
    goals = [[round(float(s["x"]), 3), round(float(s["y"]), 3), round(float(s["z"]), 3)]
             for s in plan.get("subgoals", []) if all(k in s for k in ("x", "y", "z"))]
    os.makedirs(out, exist_ok=True)
    json.dump(plan, open(os.path.join(out, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"goals": goals}, open(os.path.join(out, "goals.json"), "w"), indent=2)
    return plan, goals


def infer(instruction, rgb, grid, model=MODEL_ID, timeout=None,
          prompt="VLM_guidedGCRL_prompt.txt", goal=None, out=OUT,
          modality="both", dtype="bf16", load_4bit=False, think=False, max_new_tokens=None,
          brief=True):
    """프로그램 호출용(VLM_infer.infer 드롭인) — VLM 계획(dict) 반환 + out/plan.json, goals.json 저장.
    model: HF 모델 id. timeout은 호환용(로컬 추론엔 미사용). modality: RQ1 입력 ablation.
    think: Qwen3.5 thinking 모드(품질↑·느림). max_new_tokens None이면 think에 맞춰 자동."""
    if max_new_tokens is None:
        max_new_tokens = 8192 if think else 1536       # thinking은 추론+JSON라 넉넉히
    use_rgb = modality in ("both", "rgb_only")
    use_grid = modality in ("both", "height_only")
    for p, used in ((rgb, use_rgb), (grid, use_grid)):
        if used and not os.path.exists(p):
            raise FileNotFoundError(f"입력 이미지 없음: {p} (먼저 render_scene.py 실행)")
    messages = build_messages(instruction, os.path.abspath(rgb), os.path.abspath(grid),
                              prompt=prompt, goal=goal, modality=modality, brief=brief)
    raw = generate(messages, model_id=model, dtype=dtype, load_4bit=load_4bit,
                   max_new_tokens=max_new_tokens, think=think)
    plan, _ = parse_and_save(raw, out)
    return plan


def _parse_goal(s):
    if not s:
        return None
    v = [float(t) for t in s.replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"--goal 형식 오류: 'x,y,z' 여야 함 (받음: {s})")
    return v


def _goal_from_scene(grid):
    sp = os.path.join(os.path.dirname(os.path.abspath(grid)), "scene.json")
    if os.path.exists(sp):
        g = json.load(open(sp)).get("goal")
        return [float(v) for v in g] if g else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="주어진 목표 좌표에 장애물을 피해 도달하는 서브골 시퀀스를 계획하라.")
    ap.add_argument("--rgb", default=os.path.join(OUT, "cabin_init.png"))
    ap.add_argument("--grid", default=os.path.join(OUT, "gridmap.png"))
    ap.add_argument("--prompt", default="VLM_guidedGCRL_prompt.txt",
                    help="system prompt 파일(Experiment/Code/prompts/ 또는 VLM/ 상대, 절대경로도 가능)")
    ap.add_argument("--goal", default=None, help="목표 좌표 'x,y,z'. 생략 시 grid 폴더 scene.json에서 로드")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default=MODEL_ID, help="HF 모델 id(예: Qwen/Qwen3.5-4B|-27B|-35B-A3B|-122B-A10B)")
    ap.add_argument("--modality", default="both", choices=MODALITIES, help="RQ1 입력 ablation")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--load_4bit", action="store_true", help="대형 모델(122B 등) 4bit 적재(bitsandbytes 필요)")
    ap.add_argument("--think", action="store_true", help="Qwen3.5 thinking 모드(품질↑·느림, 토큰↑)")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="기본: think면 8192, 아니면 1536")
    a = ap.parse_args()
    goal = _parse_goal(a.goal)
    if goal is None and "guidedGCRL" in a.prompt:
        goal = _goal_from_scene(a.grid)
    plan = infer(a.instruction, a.rgb, a.grid, model=a.model, prompt=a.prompt, goal=goal,
                 out=a.out, modality=a.modality, dtype=a.dtype, load_4bit=a.load_4bit,
                 think=a.think, max_new_tokens=a.max_new_tokens)
    print("scene:", plan.get("scene", ""))
    print("objects:", [(o.get("type"), o.get("ref")) for o in plan.get("objects", [])])
    print("subgoals:", [[s.get("x"), s.get("y"), s.get("z")] for s in plan.get("subgoals", [])])
    if goal is not None:
        print("target goal:", goal)
    print(f"저장: {a.out}/plan.json, goals.json  (model={a.model}, modality={a.modality}, prompt={a.prompt})")


if __name__ == "__main__":
    main()
