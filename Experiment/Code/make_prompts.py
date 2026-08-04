"""
RQ1(모달리티)·RQ4(프롬프트 요소) ablation용 프롬프트 변형 생성기.
base = VLM/system_prompt.txt(상차 loading 프롬프트, "═══ N. 제목 ═══" 섹션 구분).
섹션을 번호로 선택/제거해 재현 가능하게 변형 파일을 Experiment/Code/prompts/에 쓴다.

  python Experiment/Code/make_prompts.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "VLM", "system_prompt.txt")
OUT = os.path.join(HERE, "prompts")
_HDR = re.compile(r"^═+\s*(\d+)\.\s*(.*?)\s*═+\s*$")


def parse_sections(text):
    """(preamble, [(num, header_line, body)]) — ═══ N. ═══ 헤더로 분할."""
    lines = text.splitlines(keepends=True)
    pre, secs, cur = [], [], None
    for ln in lines:
        m = _HDR.match(ln.strip("\n"))
        if m:
            if cur is not None:
                secs.append(cur)
            cur = [int(m.group(1)), ln, ""]
        elif cur is None:
            pre.append(ln)
        else:
            cur[2] += ln
    if cur is not None:
        secs.append(cur)
    return "".join(pre), secs


def build(preamble, secs, keep=None, drop=None, header_note="", override=None):
    """keep(집합) 지정 시 그 번호만, 아니면 drop(집합) 제외. header_note는 맨 앞에 삽입.
    override={num: (hdr, body)} 지정 시 해당 섹션을 통째로 교체(RQ1 모달리티용)."""
    override = override or {}
    out = header_note + preamble
    for num, hdr, body in secs:
        if keep is not None and num not in keep:
            continue
        if drop is not None and num in drop:
            continue
        if num in override:
            hdr, body = override[num]
        out += hdr + body
    return out


# ── RQ1 모달리티: 섹션 1~2(입력 이미지 해석·장애물 인식)만 실제 제공 이미지에 맞게 교체.
#    섹션 3~10(계획 로직·출력)은 세 모달리티 공통(통제 변수)으로 그대로 둔다.
_H1 = "═══════════════ 1. 입력 이미지 해석 ═══════════════\n"
_H2 = "═══════════════ 2. 장애물 영역 인식 규칙 ═══════════════\n"

# RGB only — grid map 미제공. 위치(θ,r)·높이(h)를 RGB 원근·겉보기 크기로 추정.
_SEC1_RGB = (
    "[이미지 1] RGB 카메라 뷰 — 굴착기 cabin 상단-우측 카메라의 전방 시점(수평 화각 135°, 약 −67°~+67°).\n"
    "  · 화면 좌우 = 방위각 θ(왼쪽 −, 중앙 0°=정면, 오른쪽 +).\n"
    "  · 물체의 겉보기 크기·화면상 높이 = 거리 r 단서(멀수록 작고 위쪽에 보임).\n"
    "  · 물체의 겉보기 세로 크기 = 높이 h 단서. h는 0~2.8 m 범위로 **보수적으로(상한 쪽) 추정**.\n"
    "  물체의 외형·색으로 종류를 판정: 흙더미(갈색 둔덕=픽업 대상), 덤프트럭(노란 차량=투하 대상), 장애물(회색 박스=회피 대상).\n\n"
    "[입력 안내] **grid map(높이맵)은 제공되지 않습니다.** 모든 물체의 위치(θ, r)와 높이(h_max)를 RGB만으로 추정해\n"
    "  §3의 (x, y, z)로 변환하세요. 정확한 수치 지도가 없으므로 원근·겉보기 크기에 근거해 최선 추정합니다.\n\n"
)
_SEC2_RGB = (
    "RGB에서 각 장애물(회색 박스)을 개별 영역으로 인식하세요. 장애물이 여러 개면 각각 독립적으로 처리합니다.\n"
    "각 장애물에 대해 RGB 원근으로 다음을 **추정**하세요(수치 지도 없음):\n"
    "  · θ_min ~ θ_max : 화면상 좌우 끝의 방위각 범위\n"
    "  · r_min ~ r_max : 겉보기 크기·화면 높이로부터의 거리 범위\n"
    "  · h_max         : 겉보기 세로 크기로부터의 높이(상한 쪽 보수적)\n"
    "  · 영역 대표 r   : (r_min + r_max) / 2 — passable 판정에 사용\n\n"
)

# Height only — RGB 미제공. 종류(흙더미/트럭/장애물)를 grid map 위치·높이 blob만으로 추정.
_SEC1_GRID = (
    "[입력 안내] **RGB 카메라 뷰(이미지 1)는 제공되지 않습니다.** 물체 종류(흙더미/트럭/장애물)를\n"
    "  아래 grid map의 **위치·높이 blob만으로** 추정하세요(예: 좌측 둔덕형 blob=흙더미, 우측 큰 blob=트럭, 경로상 박스형=장애물).\n\n"
    "[이미지 2] 3D grid map — 굴착기 기준 (방위각 θ, 거리 r) top-down 높이맵.\n"
    "  **위치·높이 수치는 이 지도를 기준으로 읽을 것.**\n"
    "  **모든 물체(흙더미·덤프트럭·장애물)가 이 지도에 높이 blob으로 나옵니다.** 각 blob의 종류(픽업할 흙더미 /\n"
    "  투하할 트럭 / 회피할 장애물)는 위치·높이로 판단하세요.\n"
    "  주의: 트럭 blob은 투하 목적지이지 회피 대상이 아닙니다.\n"
    "  · 가로축 = 방위각 θ : 왼쪽 −90° … 중앙 0°(정면/forward, 점선) … 오른쪽 +90°.\n"
    "  · 세로축 = 거리 r   : 아래 0 m(굴착기 본체) … 위 4.5 m(원거리).\n"
    "  · 색 = 지면 대비 높이 h (m), 0.4 m 단위 7밴드. 지면은 평탄 plane(구덩이/굴착 없음, 음수 높이 없음).\n"
    "    오른쪽 상단 범례 기준:\n"
    "      회색        h  0   ~  0.4  지면(평탄, 통과 가능)\n"
    "      옅은 노랑   h  0.4 ~  0.8  낮은 장애물\n"
    "      노랑        h  0.8 ~  1.2\n"
    "      주황        h  1.2 ~  1.6\n"
    "      진한 주황   h  1.6 ~  2.0\n"
    "      빨강        h  2.0 ~  2.4\n"
    "      진한 빨강   h  2.4 ~  2.8  최고 장애물\n"
    "  · 색이 애매하면 반드시 우상단 범례와 대조할 것.\n"
    "  · **h_max는 색 구간의 상한값(0.4 단위)으로 보수적으로 읽을 것.**\n"
    "    하한값이나 중간값을 사용하면 충돌이 발생한다.\n"
    "    예: 노랑(0.8~1.2) → h_max = 1.2 / 진한 주황(1.6~2.0) → h_max = 2.0\n\n"
)


def main():
    os.makedirs(OUT, exist_ok=True)
    text = open(BASE, encoding="utf-8").read()
    pre, secs = parse_sections(text)
    nums = [s[0] for s in secs]
    print(f"base 섹션 번호: {nums}")

    files = {}
    # ── RQ4: 프롬프트 요소 ablation ──
    files["rq4_full.txt"] = build(pre, secs)                                  # 전체(=base)
    files["rq4_no_kinematics.txt"] = build(pre, secs, drop={4, 5})            # 운동학 z_max표 + passable 제거
    files["rq4_no_cot.txt"] = build(pre, secs, drop={7, 9})                   # 경로안전 3단계 + 수행순서(CoT) 제거
    files["rq4_no_clearance.txt"] = build(pre, secs, drop={6})               # 안전여유 규칙 제거
    files["rq4_minimal.txt"] = build(pre, secs, keep={3, 8, 10})             # 좌표계+태스크+출력형식만

    # ── RQ1: 모달리티(실제 이미지 1장 + 섹션1~2를 제공 이미지에 맞게 교체) ──
    #   섹션 3~10은 세 모달리티 공통(통제 변수). 섹션 1(입력 해석)·2(장애물 인식)만 교체.
    files["rq1_both.txt"] = build(pre, secs)                                  # 두 이미지(=base)
    files["rq1_rgb_only.txt"] = build(pre, secs, override={                   # RGB만: grid 설명 제거
        1: (_H1, _SEC1_RGB), 2: (_H2, _SEC2_RGB)})
    files["rq1_height_only.txt"] = build(pre, secs, override={               # gridmap만: RGB 설명 제거
        1: (_H1, _SEC1_GRID)})

    for name, content in files.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  wrote {name}  ({len(content)} chars)")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
