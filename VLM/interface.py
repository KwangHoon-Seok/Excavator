# VLM의 출력을 GCRL의 인풋에 활용할 수 있도록 변환하는 모듈
"""
VLM 출력(plan.json) → GCRL 입력 어댑터.
학습된 GCRL 모델 추론 시 사용: VLM이 준 subgoal 시퀀스·장애물을 GCRL이 먹는 형식으로 변환·주입.

역할(변환만):
  - subgoals(x,y,z 순서)          → goal 시퀀스 [[x,y,z], ...]  (relay용)
  - objects(type="obstacle")     → 축정렬 박스(center, half)  → GCRL env에 배치
좌표계 일치: VLM (x=우, y=전방≥0, z=높이) = GCRL 월드. θ=arctan2(x,y)(전방=0).

사용 예 (추론 루프):
    from interface import parse_plan, apply_obstacles
    p = parse_plan("VLM/out/plan.json")
    env.reset(options={"goal": p["goals"][0]})
    apply_obstacles(env, p["obstacles"]); mujoco.mj_forward(env.model, env.data)
    # 이후 p["goals"]를 순차로 set하며 policy 실행(relay)
"""
import json
import numpy as np


def _polar_to_box(o):
    """장애물 극좌표 sector(r_range, theta_range, h_max) → 축정렬 박스(center[x,y,z], half).
    바닥 z=0에 붙임(center z=h/2). sector의 Cartesian bounding box로 근사. θ는 deg, x=r·sinθ, y=r·cosθ."""
    h = float(o.get("h_max", o.get("z", 0.5)) or 0.5)
    if "r_range" in o and "theta_range" in o:
        rmin, rmax = (float(v) for v in o["r_range"])
        thmin, thmax = (np.radians(float(v)) for v in o["theta_range"])
        pts = [(r * np.sin(t), r * np.cos(t)) for r in (rmin, rmax) for t in (thmin, thmax)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        hx, hy = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
    elif "x" in o and "y" in o:                       # 폴백: 중심만 있으면 기본 크기
        cx, cy, hx, hy = float(o["x"]), float(o["y"]), 0.4, 0.4
    else:
        return None
    return {"center": [cx, cy, h / 2.0],
            "half": [max(hx, 0.15), max(hy, 0.15), max(h / 2.0, 0.1)]}


def parse_plan(plan):
    """plan(dict 또는 plan.json 경로) → {"goals":[[x,y,z]...], "obstacles":[{center,half,ref}...]}."""
    if isinstance(plan, str):
        plan = json.load(open(plan, encoding="utf-8"))
    goals = [[float(s["x"]), float(s["y"]), float(s["z"])]
             for s in plan.get("subgoals", []) if all(k in s for k in ("x", "y", "z"))]
    obstacles = []
    for o in plan.get("objects", []):
        if o.get("type") != "obstacle":
            continue
        box = _polar_to_box(o)
        if box:
            obstacles.append({**box, "ref": o.get("ref", "")})
    return {"goals": goals, "obstacles": obstacles}


def apply_obstacles(env, obstacles):
    """VLM 장애물을 GCRL env에 배치(랜덤 _place_obstacles 대신). env._obs_geoms에 위치·크기 세팅 +
    _obs_sph(캡슐 TTC용 내접구)·_active_obs(버킷 mj_geomDistance용) 갱신 → observation·보상에 반영.
    env는 obstacle_training=True로 생성돼 있어야 함. 장애물 수는 env._n_obs_geom(기본 4)까지."""
    m = env.model
    env._obs_sph = []; env._active_obs = []
    for i, (ob, og) in enumerate(zip(env._obs_bodies, env._obs_geoms)):
        if i < len(obstacles):
            c = np.asarray(obstacles[i]["center"], float)
            half = np.asarray(obstacles[i]["half"], float)
            m.body_pos[ob] = c
            m.geom_size[og] = half
            centers, R = env._box_spheres(c, half)
            env._obs_sph.extend((cc, R) for cc in centers)
            env._active_obs.append(og)
        else:
            m.body_pos[ob] = [0, 0, 50.0]              # 미사용 슬롯은 멀리
    return len(env._active_obs)
