"""
미션 challenge seed 스크리닝(기하 기준, 컨트롤러 무관 → 편향 없음).
HOME→pile 또는 pile→truck 직선 corridor 안에 장애물이 실제로 놓인 seed만 채택
(=회피가 필요한 진짜 도전 씬). 결과 seeds_mission.json 저장.

  python Experiment/Code/screen_mission_seeds.py --need 100
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mission as M                                             # noqa: E402
sys.path.insert(0, os.path.join(M.ROOT, "VLM"))
import mission_scene as MS                                      # noqa: E402

HOME_XY = np.array(MS._HOME_TIP_XY)
PILE_XY = np.array(MS.PILE_XY)
TRUCK_XY = np.array(MS.TRUCK_XY)


def _seg_dist(p, a, b):
    """xy 점 p에서 선분 ab까지 최단거리."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def is_challenge(seed, corridor=0.6, dist="id"):
    """HOME→pile 또는 pile→truck corridor(+장애물 반경) 안에 장애물이 있으면 challenge."""
    for o in M.gen_obstacles(seed, dist):
        c = np.array(o["center"][:2]); rad = max(o["half"][:2])
        d = min(_seg_dist(c, HOME_XY, PILE_XY), _seg_dist(c, PILE_XY, TRUCK_XY))
        if d < corridor + rad:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=100)
    ap.add_argument("--scan", type=int, default=1000, help="최대 스캔 seed 수")
    ap.add_argument("--corridor", type=float, default=0.6)
    ap.add_argument("--dist", default="id", choices=["id", "ood"])
    ap.add_argument("--out", default=os.path.join(HERE, "seeds_mission.json"))
    a = ap.parse_args()
    seeds = []
    for s in range(a.scan):
        if is_challenge(s, a.corridor, a.dist):
            seeds.append(s)
        if len(seeds) >= a.need:
            break
    json.dump(seeds, open(a.out, "w"))
    print(f"채택 {len(seeds)}/{a.need}  (스캔 {s+1} seeds, corridor={a.corridor})  → {a.out}")
    print("seeds[:20]:", seeds[:20])


if __name__ == "__main__":
    main()
