"""
VLM 임무계획 추론 — Claude(claude CLI)로 RGB + gridmap → 서브골 시퀀스 JSON.
로컬 Qwen(infer.py)의 대체판: PC 사양 부족으로 VLM=Claude. 별도 API 키 불필요
(claude CLI가 기존 Claude Code 로그인 인증 사용). 모델 버전은 --model로 선택.

입력: RGB(cabin_init.png) + gridmap.png + system prompt(--prompt) (+ 같은 폴더 gridmap_legend.txt)
출력: plan.json(scene/objects/subgoals), goals.json(GCRL용 goal 리스트) — --out 폴더에 저장
프롬프트 2종:
  - system_prompt.txt          : 상차(pile→truck) loading 임무
  - VLM_guidedGCRL_prompt.txt  : **주어진 목표 좌표 도달 + 장애물 회피**(A/B 실험용, --goal 필요)
실행:
  loading:  conda run -n seok python VLM/VLM_infer.py --instruction "왼쪽 흙더미를 퍼서 오른쪽 트럭에 실어라" --model sonnet
  reach  :  conda run -n seok python VLM/VLM_infer.py --prompt VLM_guidedGCRL_prompt.txt \\
              --rgb .../seed3/cabin_init.png --grid .../seed3/gridmap.png --out .../seed3 --goal 1.2,3.4,0.3
            (--goal 생략 시 grid 폴더의 scene.json에서 자동 로드)
"""
import os
import re
import json
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def _resolve_prompt(prompt):
    """프롬프트 파일 경로 해석 — 절대경로면 그대로, 아니면 VLM/ 기준 상대."""
    return prompt if os.path.isabs(prompt) else os.path.join(HERE, prompt)


def build_prompt(instruction, rgb, grid, prompt="system_prompt.txt", goal=None, legend=None):
    """system prompt + (legend) + (goal 좌표) + 이미지 경로 + 지시. claude가 Read 툴로 이미지를 읽는다.
    goal: [x,y,z] 주면 '목표 좌표' 줄을 삽입(reach 임무). legend: 색↔높이 텍스트 파일 경로(없으면 grid 옆 자동)."""
    sp = open(_resolve_prompt(prompt), encoding="utf-8").read()
    if legend is None:
        legend = os.path.join(os.path.dirname(os.path.abspath(grid)), "gridmap_legend.txt")
    if legend and os.path.exists(legend):
        sp += "\n\n[grid map 색↔높이 legend]\n" + open(legend, encoding="utf-8").read()
    goal_line = ""
    if goal is not None:
        goal_line = (f"목표 좌표(버킷 끝 도달점): ({float(goal[0]):.3f}, {float(goal[1]):.3f}, "
                     f"{float(goal[2]):.3f}) [m]. 이 좌표가 최종 서브골이며, 장애물을 피해 도달하라.\n")
    sp += (f"\n\n═══ 입력 이미지 (Read 툴로 읽어라) ═══\n"
           f"[이미지 1] RGB cabin view (장애물 식별용): {rgb}\n"
           f"[이미지 2] 3D grid map (장애물 위치·높이, top-down 극좌표): {grid}\n"
           f"{goal_line}"
           f"지시: {instruction}\n\n"
           f"위 두 이미지를 Read로 읽고, 섹션 10 형식의 JSON만 출력하라. JSON 밖의 텍스트/마크다운 금지.")
    return sp


def call_claude(prompt, model, timeout=900):
    """claude CLI headless(-p) 호출 → JSON envelope의 result(모델 텍스트) 반환.
    주의: 에이전트가 이미지 2개+긴 프롬프트+clearance 추론을 하므로 scene당 수 분 걸릴 수 있음."""
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # TimeoutError(=Exception)로 던져야 배치 러너의 except Exception이 잡아 fallback 가능
        # (SystemExit은 BaseException이라 안 잡혀 전체 실행이 크래시함)
        raise TimeoutError(f"claude CLI 타임아웃({timeout}s) — --timeout 늘리거나 --model을 빠른 것으로.")
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI 실패(exit {r.returncode}): {r.stderr[:600]}")
    try:
        env = json.loads(r.stdout)
    except json.JSONDecodeError:
        return r.stdout                        # 이미 순수 텍스트면 그대로
    return env.get("result", "")


def parse_and_save(raw, out=OUT):
    """모델 텍스트에서 JSON 추출 → out/plan.json, goals.json 저장. (result, goals) 반환."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise SystemExit("JSON을 찾지 못함 — 원본 출력:\n" + raw[:800])
    result = json.loads(m.group())
    goals = [[round(float(s["x"]), 3), round(float(s["y"]), 3), round(float(s["z"]), 3)]
             for s in result.get("subgoals", []) if all(k in s for k in ("x", "y", "z"))]
    os.makedirs(out, exist_ok=True)
    json.dump(result, open(os.path.join(out, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"goals": goals}, open(os.path.join(out, "goals.json"), "w"), indent=2)
    return result, goals


def infer(instruction, rgb, grid, model="sonnet", timeout=900,
          prompt="system_prompt.txt", goal=None, out=OUT):
    """프로그램에서 직접 호출용: VLM 계획(dict) 반환 + out/plan.json, goals.json 저장.
    prompt: system prompt 파일(상차=system_prompt.txt / reach=VLM_guidedGCRL_prompt.txt).
    goal:  [x,y,z] 주면 목표 좌표 도달 임무로 프롬프트에 삽입."""
    for p in (rgb, grid):
        if not os.path.exists(p):
            raise SystemExit(f"입력 이미지 없음: {p} (먼저 render_scene.py/make_gridmap.py 실행)")
    raw = call_claude(build_prompt(instruction, os.path.abspath(rgb), os.path.abspath(grid),
                                   prompt=prompt, goal=goal), model, timeout)
    result, _ = parse_and_save(raw, out)
    return result


def _parse_goal(s):
    """'x,y,z' 문자열 → [x,y,z] float. None/빈 문자열이면 None."""
    if not s:
        return None
    v = [float(t) for t in s.replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"--goal 형식 오류: 'x,y,z' 여야 함 (받음: {s})")
    return v


def _goal_from_scene(grid):
    """--goal 미지정 시 grid 폴더의 scene.json에서 goal 자동 로드(reach 임무 편의)."""
    sp = os.path.join(os.path.dirname(os.path.abspath(grid)), "scene.json")
    if os.path.exists(sp):
        g = json.load(open(sp)).get("goal")
        return [float(v) for v in g] if g else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="장면을 분석해 서브골 시퀀스를 계획하라.")
    ap.add_argument("--rgb", default=os.path.join(OUT, "cabin_init.png"))
    ap.add_argument("--grid", default=os.path.join(OUT, "gridmap.png"))
    ap.add_argument("--prompt", default="system_prompt.txt",
                    help="system prompt 파일. 상차=system_prompt.txt / 목표도달+회피=VLM_guidedGCRL_prompt.txt")
    ap.add_argument("--goal", default=None,
                    help="목표 좌표 'x,y,z'(reach 임무). 생략 시 grid 폴더 scene.json에서 자동 로드")
    ap.add_argument("--out", default=OUT, help="plan.json/goals.json 저장 폴더(기본 VLM/out)")
    ap.add_argument("--model", default="sonnet",
                    help="claude 모델 버전(예: haiku, sonnet, opus, 또는 정확한 id claude-sonnet-5 등)")
    ap.add_argument("--timeout", type=int, default=900, help="claude 호출 타임아웃(초). scene당 수 분 걸릴 수 있음")
    a = ap.parse_args()
    goal = _parse_goal(a.goal)
    if goal is None and "guidedGCRL" in a.prompt:      # reach 프롬프트인데 goal 없으면 scene.json 시도
        goal = _goal_from_scene(a.grid)
    result = infer(a.instruction, a.rgb, a.grid, a.model, a.timeout,
                   prompt=a.prompt, goal=goal, out=a.out)
    print("scene:", result.get("scene", ""))
    print("objects:", [(o.get("type"), o.get("ref")) for o in result.get("objects", [])])
    goals = [[s.get("x"), s.get("y"), s.get("z")] for s in result.get("subgoals", [])]
    print("subgoals:", goals)
    if goal is not None:
        print("target goal:", goal)
    print(f"저장: {a.out}/plan.json, goals.json  (프롬프트={a.prompt}, 모델={a.model})")


if __name__ == "__main__":
    main()
