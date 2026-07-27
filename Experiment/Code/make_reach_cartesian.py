"""VLM_guidedGCRL_prompt.txt(reach+avoid, 극좌표) → Cartesian 버전 생성.
§1 이미지 축(θ,r → x,y) / §2 장애물 범위(θ,r → x,y) / §3 극좌표 변환 제거만 교체.
§4~7 운동학·clearance·경로검사는 r=√(x²+y²), θ=atan2(x,y)를 x,y에서 유도하므로 유지."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "VLM", "VLM_guidedGCRL_prompt.txt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "reach_cartesian.txt")

REPL = [
    # §1 [이미지 2] 축 설명: 극좌표 → 직교
    ("""[이미지 2] 3D grid map — 굴착기 기준 (방위각 θ, 거리 r) top-down 높이맵. **장애물의 위치·높이 기준.**
  **위치·높이 수치는 반드시 이 지도를 기준으로 읽을 것. RGB 이미지로 거리·높이를 추정하지 말 것.**
  · 가로축 = 방위각 θ : 왼쪽 −90° … 중앙 0°(정면/forward, 점선) … 오른쪽 +90°.
  · 세로축 = 거리 r   : 아래 0 m(굴착기 본체) … 위 4.5 m(원거리).""",
     """[이미지 2] 3D grid map — 굴착기 기준 top-down 높이맵(**직교좌표 x, y**). **장애물의 위치·높이 기준.**
  **위치·높이 수치는 반드시 이 지도를 기준으로 읽을 것. RGB 이미지로 거리·높이를 추정하지 말 것.**
  · 가로축 = x(좌우, m) : 왼쪽 −(음수) … 중앙 0(굴착기 정면 축, 점선) … 오른쪽 +(양수).
  · 세로축 = y(전방거리, m) : 아래 0 m(굴착기 본체) … 위 4.5 m(원거리). 항상 y ≥ 0.
  · 지도에서 각 물체의 (x, y)[m]를 **직접** 읽습니다(θ,r 극좌표 변환 불필요)."""),
    # §2 장애물 항목: θ/r 범위 → x/y 범위
    ("""  · θ_min ~ θ_max : 색칠된 셀의 좌우 끝 (방위각 범위)
  · r_min ~ r_max : 색칠된 셀의 상하 끝 (거리 범위)
  · h_max         : 색 구간 상한값 (위 규칙 적용)
  · 영역 대표 r   : (r_min + r_max) / 2 — passable 판정에 사용""",
     """  · x_min ~ x_max : 색칠된 셀의 좌우 끝 (x 범위, m)
  · y_min ~ y_max : 색칠된 셀의 상하 끝 (y 범위, m)
  · h_max         : 색 구간 상한값 (위 규칙 적용)
  · 영역 중심     : ((x_min+x_max)/2, (y_min+y_max)/2). 대표 거리 r = √(중심x² + 중심y²) — passable 판정에 사용"""),
    # §3 극좌표 변환 라인 제거
    ("  변환: x = r·sin(θ),  y = r·cos(θ),  z = h   (θ는 도 단위)",
     "  grid map에서 x, y를 직접 읽고 z = h. (극좌표 변환 불필요.)"),
    # §10 출력 스키마: theta_range/r_range → x_range/y_range (실행엔 미사용, VLM 추론 일관성용)
    ('"theta_range": [<θ_min>, <θ_max>],\n      "r_range": [<r_min>, <r_max>],',
     '"x_range": [<x_min>, <x_max>],\n      "y_range": [<y_min>, <y_max>],'),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t = open(SRC, encoding="utf-8").read()
    for a, b in REPL:
        if a not in t:
            print("⚠️ 미매칭(스킵):", a[:40], "...")
        t = t.replace(a, b)
    # 나머지 문구의 '방위각 θ' 잔재를 x로 완화(경로검사 §7의 θ_k는 atan2로 계산되므로 유지)
    open(OUT, "w", encoding="utf-8").write(t)
    print("wrote", OUT, f"({len(t)} chars)")


if __name__ == "__main__":
    main()
