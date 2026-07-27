"""
상차(loading) 미션 하네스 — 흙더미(pile) → 트럭(truck), 장애물 회피.

이 실험의 환경/실행 규약:
  · 씬 렌더(VLM 입력): mission_scene(흙더미+트럭+장애물) MjModel에서 cabin RGB + 극좌표 gridmap.
  · GCRL 실행: ExcavatorRHEREnv(obstacle_training=True)에 **씬의 실제 장애물**을 interface.apply_obstacles로
    주입(GT). 흙더미·트럭은 목표 좌표(PILE_PICK/TRUCK_DUMP)로만 사용(GCRL은 waypoint reacher).
  · 미션 성공 = 흙더미 픽업점 도달(stage1) AND 트럭 투하점 도달(stage2) AND 충돌 0(장애물 min_gap≤0 없음).

ID/OOD(RQ3): 장애물 분포축을 이동. ID=학습친화(낮은 장애물, 위로 통과 가능),
             OOD=분포 밖(더 높고/크고/많은 장애물 → 반드시 수평 우회, 일부 학습 상한 초과).
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "Model", "Version4"))     # excavator_env(HOME)
sys.path.insert(0, os.path.join(ROOT, "VLM"))                   # gridmap, mission_scene, interface
sys.path.insert(0, os.path.join(ROOT, "Evaluate", "vlm_guided"))  # render_scene(build_env)

import gridmap                                                  # noqa: E402
import mission_scene as MS                                      # noqa: E402
import interface                                                # noqa: E402
import render_scene                                             # noqa: E402
from excavator_env import HOME                                  # noqa: E402

# 픽업점 z: mission_scene 기본은 peak-0.15=0.85지만, v4base 컨트롤러가 먼 좌측 저고도(r≈3.35,z0.85)에서
# 도달 실패(능력 절벽) → 흙더미 정상 z=1.0으로 상향(직접 도달 ~0.19, tol 내). 트럭 투하점은 그대로.
PILE_PICK = [MS.PILE_XY[0], MS.PILE_XY[1], 1.0]
# 트럭 투하점 = 트럭 적재함(배드) 중심. MS.TRUCK_DUMP(3.2,2.6)은 트럭 배치원점(모서리)이라
# 큰 트럭 블롭 중심(2.35,2.58)에서 0.85m 벗어나 VLM이 트럭을 정확히 봐도 도달 실패했음 → 배드중심으로 정정.
TRUCK_DUMP = [2.35, 2.58, 1.6]
# stage 성공 판정: **수평(xy) 도달** 기준(작업레벨). 이 GCRL은 실제 scoop 물리가 없는 reacher라
# "버킷이 흙더미/트럭 footprint 위(수평)에 도달"을 완수로 본다. VLM이 높이(z)를 보수적으로 높게 읽는
# 경향이 있어 z를 성공조건에서 제외(수평 도달 + 충돌0 = safe_success). footprint 반경+컨트롤러 정밀도 감안 0.85m.
STAGE_TOL_XY = 0.85
REACH_TOL = 0.60          # waypoint 전환(3D) 허용오차(plan 추종용)
# 트럭 도달(stage2) = 버킷 tip이 트럭 적재함 footprint 안(수평)에 들어오면 성공.
# 트럭은 큰 물체라 '점 tol'이 아니라 footprint로 판정(장애물없는 씬에서 측정한 트럭 top 범위 + 버킷 여유).
TRUCK_FOOTPRINT = (1.57, 3.89, 2.04, 3.32)   # (x_min, x_max, y_min, y_max)
TRUCK_FP_MARGIN = 0.20                        # 버킷 크기·도달 여유
INSTRUCTION = "흙더미로 이동 후, 장애물을 회피하며 트럭으로 이동해라"


def _in_truck_fp(xy, fp=TRUCK_FOOTPRINT, margin=TRUCK_FP_MARGIN):
    x0, x1, y0, y1 = fp
    return (x0 - margin <= xy[0] <= x1 + margin) and (y0 - margin <= xy[1] <= y1 + margin)

# 분포 파라미터: ID=낮음·적음(위로 통과 가능) / OOD=높음·큼·**많음**(밀집 장애물 필드, 반드시 우회).
#   prox=흙더미→트럭 경로에서 배치 허용 폭(클수록 넓은 corridor→더 많이 배치), omargin=장애물 간 간격 여유.
DIST_PARAMS = {
    "id":  dict(k_range=(2, 4), hgt=MS.MISSION_HGT_RANGE, lr=MS._CFG["obs_len_range"],
                wr=MS._CFG["obs_wid_range"], prox=1.2, omargin=0.2),
    "ood": dict(k_range=(6, 8), hgt=(1.0, 2.0), lr=(0.4, 1.1), wr=(0.4, 1.1),
                prox=2.2, omargin=0.03),
    # confusable: 장애물을 흙더미(박스)와 같은 크기·높이로 → 높이맵에선 흙더미↔장애물 구별 불가(색만 다름)
    "confusable": dict(k_range=(4, 6), hgt=(0.8, 1.2), lr=(0.6, 1.0), wr=(0.6, 1.0),
                       prox=1.4, omargin=0.15),
    # tall: 장애물을 수직 통과 불가 높이(2.0~2.6 m)로 + 회랑을 좁게 → "들어올려 넘기"가 통하지 않아
    #   passable 판정·수평 우회 지식이 실제로 필요해짐(RQ4 safety ablation을 binding 시키는 조건).
    #   id는 h 0.8~1.2라 z만 올리면 전부 해결돼 안전 규칙이 무효했음. 높이 상한 2.6은 gridmap 7밴드(≤2.8) 내.
    #   id와 개수·크기·회랑 폭은 동일하고 (a) 높이 2.0~2.6 m, (b) 1개는 P→T 선분 위(onpath=1)만 다름.
    #   (b) 없이는 직진 경로가 애초에 막히지 않아(차단 26%) 우회 지식이 여전히 binding되지 않았음.
    "tall": dict(k_range=(2, 4), hgt=(2.0, 2.6), lr=MS._CFG["obs_len_range"],
                 wr=MS._CFG["obs_wid_range"], prox=1.2, omargin=0.2, onpath=1),
}


def _box_xml(name, cx, cy, half, rgba):
    """시각 전용(contype=0) 색 박스 body. top이 2*half_z 높이(바닥에서)."""
    hx, hy, hz = half
    return (f'    <body name="{name}" pos="{cx} {cy} {hz}">\n'
            f'      <geom name="{name}_g" type="box" size="{hx} {hy} {hz}" rgba="{rgba}" '
            f'contype="0" conaffinity="0"/>\n    </body>\n')

# 시작(HOME) 굴착기 충돌 방지용 기하: 몸체(프레임/트랙, 원점) + 전방으로 뻗은 팔 선분(x≈0, y 0.5→tip).
_BODY_XY = np.array([0.0, 0.0])
_ARM_BASE = np.array([0.0, 0.5])
_ARM_TIP = np.array(MS._HOME_TIP_XY)   # (0, 3.62)

# ── 랜덤 배치(RQ 확장): pile/truck을 seed마다 아무 위치(양쪽 어디든)로 → 위치기반 분류 차단 ──
# 트럭 mesh footprint·투하점의 '배치원점 대비 상대 오프셋'(고정 트럭(3.2,2.6) 측정값). 평행이동해 per-seed 계산.
TRUCK_QUAT = MS.TRUCK_QUAT                       # 기본 heading
TRUCK_QUAT_FLIP = "-0.707 0 0 0.707"            # 180° about z(반대 heading) → footprint 좌우 반전
_TRUCK_DUMP_REL = (-0.85, -0.02)               # 배드중심 − 트럭 배치원점(기본 heading)
_TRUCK_FP_REL = (-1.63, 0.69, -0.56, 0.72)     # footprint 상대 (dx0, dx1, dy0, dy1)


def pile_pick_of(pile_xy):
    return [float(pile_xy[0]), float(pile_xy[1]), 1.0]


def truck_dump_of(truck_xy, flip=False):
    dx, dy = (-_TRUCK_DUMP_REL[0], -_TRUCK_DUMP_REL[1]) if flip else _TRUCK_DUMP_REL
    return [truck_xy[0] + dx, truck_xy[1] + dy, 1.6]


def truck_fp_of(truck_xy, flip=False):
    dx0, dx1, dy0, dy1 = _TRUCK_FP_REL
    if flip:                                                            # 180° 회전 → (dx,dy)→(−dx,−dy)
        dx0, dx1, dy0, dy1 = -dx1, -dx0, -dy1, -dy0
    return (truck_xy[0] + dx0, truck_xy[0] + dx1, truck_xy[1] + dy0, truck_xy[1] + dy1)


def sample_targets(seed):
    """seed별 pile/truck xy 랜덤(양쪽 어디든, 역할·트럭heading 랜덤). → (pile_xy, truck_xy, truck_flip)."""
    rng = np.random.default_rng(int(seed) * 99991 + 7)
    H = np.array(MS._HOME_TIP_XY)
    for _ in range(4000):
        def samp():
            r = rng.uniform(2.8, 3.4); th = np.deg2rad(rng.uniform(-70, 70))
            return np.array([r * np.sin(th), r * np.cos(th)])
        A, B = samp(), samp()
        P, T = (A, B) if rng.random() < 0.5 else (B, A)                 # 역할 랜덤 스왑
        if np.linalg.norm(P - T) < 3.4:                                 # 분리
            continue
        if abs(np.degrees(np.arctan2(P[0], P[1]) - np.arctan2(T[0], T[1]))) < 45:
            continue
        if min(np.linalg.norm(P - H), np.linalg.norm(T - H)) < 1.2:     # home 회피
            continue
        # 트럭 heading: footprint가 작업반경 안에 드는 쪽 선택(둘 다 되면 랜덤)
        ok = []
        for flip in (False, True):
            fp = truck_fp_of(T, flip)
            rr = [np.hypot(x, y) for x, y in [(fp[0], fp[2]), (fp[1], fp[2]), (fp[0], fp[3]), (fp[1], fp[3])]]
            if max(rr) <= 4.35 and min(rr) >= 1.2:
                ok.append(flip)
        if not ok:
            continue
        flip = ok[0] if len(ok) == 1 else bool(rng.random() < 0.5)
        return (float(P[0]), float(P[1])), (float(T[0]), float(T[1])), flip
    return tuple(MS.PILE_XY), tuple(MS.TRUCK_XY), False                 # 폴백


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _gen_obstacles(seed, k_range, hgt, lr, wr, prox=1.2, omargin=0.2, P=None, T=None, onpath=0):
    """장애물을 **흙더미→트럭 traverse 경로**(선분 위 t + 수직 offset, 임의 P/T에 대응)에 배치.
    HOME→흙더미 접근로는 비움. pile/truck/home/작업반경 회피. 결정론적 seed.
    onpath>0이면 처음 onpath개는 P→T 선분 **위**(수직 offset ≈0, 중앙부)에 배치 →
    직진 경로가 반드시 막힘(안전 계획 지식이 binding되는 조건. RQ4-E2)."""
    rng = np.random.default_rng(int(seed) + 10007)
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    P = np.array(MS.PILE_XY if P is None else P, float)
    T = np.array(MS.TRUCK_XY if T is None else T, float)
    H = np.array(MS._HOME_TIP_XY)
    avoid = [(P, 0.7), (T, 1.3), (H, 0.7)]
    seg = T - P; L = float(np.linalg.norm(seg)) + 1e-9; nrm = np.array([-seg[1], seg[0]]) / L
    obs = []
    n_try = k * 400 if onpath else k * 200
    for i in range(n_try):
        if len(obs) >= k:
            break
        half = np.array([rng.uniform(*lr), rng.uniform(*wr), rng.uniform(*hgt)]) / 2.0
        if len(obs) < onpath and i < 3 * n_try // 4:        # 선분 위(막는 장애물)
            off = 0.15 if i < n_try // 2 else 0.35          # 후반부는 offset 허용범위 완화
            #  마지막 1/4 구간에서는 일반 배치로 폴백(선분 위 배치 불가능한 기하에서 0개가 되지 않도록)
            c_xy = P + rng.uniform(0.35, 0.65) * seg + rng.uniform(-off, off) * nrm
        else:
            c_xy = P + rng.uniform(0.15, 0.85) * seg + rng.uniform(-prox, prox) * nrm  # P→T corridor 내
        c = np.array([c_xy[0], c_xy[1], half[2]])
        rad = float(max(half[0], half[1]))
        rr = float(np.hypot(c_xy[0], c_xy[1]))
        if rr < 1.3 or rr > 4.3:                            # 작업반경 안
            continue
        if _seg_dist(c_xy, H, P) < 0.6 + rad:               # HOME→흙더미 접근로 비움
            continue
        if np.linalg.norm(c_xy - _BODY_XY) < 1.1 + rad:     # 몸체 회피
            continue
        if _seg_dist(c_xy, _ARM_BASE, _ARM_TIP) < 0.5 + rad:  # 시작 팔 회랑
            continue
        if any(np.linalg.norm(c[:2] - a) < rad + clr for a, clr in avoid):
            continue
        if any(np.linalg.norm(c[:2] - np.array(o["center"][:2])) < rad + max(o["half"][:2]) + omargin for o in obs):
            continue
        obs.append({"center": [round(float(v), 3) for v in c], "half": [round(float(v), 3) for v in half]})
    return obs


def gen_obstacles(seed, dist="id", P=None, T=None):
    return _gen_obstacles(seed, P=P, T=T, **DIST_PARAMS[dist])


def build_mission_model(obstacles, pile_xy=None, truck_xy=None, truck_flip=False, confusable=False):
    """흙더미+트럭 + 주어진 장애물 주입 MjModel. pile_xy/truck_xy 주면 그 위치, truck_flip이면 반대 heading.
    confusable=True면 흙더미=갈색 박스·트럭=노랑 박스(둘 다 flat 박스)로 렌더 → 높이맵에선 장애물과 형상 동일."""
    pile_xy = MS.PILE_XY if pile_xy is None else pile_xy
    truck_xy = MS.TRUCK_XY if truck_xy is None else truck_xy
    tquat = TRUCK_QUAT_FLIP if truck_flip else TRUCK_QUAT
    if confusable:
        fp = truck_fp_of(truck_xy, truck_flip)
        pile_x = _box_xml("pile", pile_xy[0], pile_xy[1], (0.4, 0.4, 0.5), "0.42 0.30 0.18 1")  # 갈색 flat 박스
        truck_x = _box_xml("truck", (fp[0]+fp[1])/2, (fp[2]+fp[3])/2,
                           ((fp[1]-fp[0])/2, (fp[3]-fp[2])/2, 0.8), "0.85 0.72 0.20 1")          # 노랑 flat 박스(footprint)
    else:
        pile_x = MS._pile_xml(*pile_xy); truck_x = MS._truck_xml(*truck_xy, quat=tquat)
    xml = open(MS.XML).read()
    # from_xml_string은 상대 meshdir을 CWD 기준으로 찾으므로 절대경로로 치환(실행 위치 무관)
    xml = xml.replace('meshdir="excavatorURDF"', 'meshdir="%s"' % os.path.join(ROOT, "excavatorURDF"))
    xml = xml.replace("</asset>", MS._asset_xml() + "  </asset>")
    xml = xml.replace("</worldbody>", pile_x + truck_x
                      + MS._obstacles_xml(obstacles) + "  </worldbody>")
    return mujoco.MjModel.from_xml_string(xml)


def render_mission(seed, out, dist="id", w=1280, h=480, fovx=135.0, n_r=45, n_theta=90, r_max=4.5,
                   randomize=False, confusable=False):
    """미션 씬 렌더 → cabin RGB + Cartesian gridmap + legend + scene.json. 반환 scene dict.
    randomize=True면 pile/truck seed별 랜덤. confusable=True면 흙더미·트럭을 박스로+장애물 크기 동일(높이맵 구별 불가)."""
    import json
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    if randomize:
        pile_xy, truck_xy, truck_flip = sample_targets(seed)
    else:
        pile_xy, truck_xy, truck_flip = tuple(MS.PILE_XY), tuple(MS.TRUCK_XY), False
    pile_pick = pile_pick_of(pile_xy); truck_dump = truck_dump_of(truck_xy, truck_flip)
    truck_fp = truck_fp_of(truck_xy, truck_flip)
    obstacles = gen_obstacles(seed, "confusable" if confusable else dist, P=pile_xy, T=truck_xy)
    m = build_mission_model(obstacles, pile_xy, truck_xy, truck_flip, confusable=confusable)
    d = mujoco.MjData(m)
    d.qpos[:4] = HOME
    mujoco.mj_forward(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cabin_cam")
    m.cam_fovy[cid] = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(fovx) / 2) / (w / h)))
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera="cabin_cam")
    rgb_path = os.path.join(out, "cabin_init.png")
    Image.fromarray(r.render()).save(rgb_path)
    r.close()

    grid = gridmap.polar_height_grid(m, d, n_r=n_r, n_theta=n_theta, r_max=r_max,
                                     object_bodies=MS.gridmap_bodies(obstacles), ground_z=gridmap.GROUND_Z)
    from render_cartesian import render_cart_from_grid                       # Cartesian top-down 뷰
    gridmap.render_grid(grid, os.path.join(out, "gridmap_polar.png"), px=(w, h))  # 참고용 극좌표
    grid_path = render_cart_from_grid(grid, os.path.join(out, "gridmap.png"))     # VLM 입력=Cartesian
    gridmap.save_grid(grid, os.path.join(out, "gridmap.npz"), os.path.join(out, "gridmap.json"))
    # 색↔높이 legend(표현 무관). 축 설명은 프롬프트가 담당.
    open(os.path.join(out, "gridmap_legend.txt"), "w").write(
        "색 = 지면 대비 높이 h(m), 0.4 m 단위 밴드: 회색 0~0.4(지면) / 옅은노랑 0.4~0.8 / "
        "노랑 0.8~1.2 / 주황 1.2~1.6 / 진주황 1.6~2.0 / 빨강 2.0~2.4 / 진빨강 2.4~2.8. "
        "h_max는 색 구간 상한값으로 보수적으로 읽을 것.")

    scene = {"seed": int(seed), "dist": dist, "pile_pick": pile_pick, "truck_dump": truck_dump,
             "truck_fp": list(truck_fp), "pile_xy": list(pile_xy), "truck_xy": list(truck_xy),
             "obstacles": obstacles, "rgb": rgb_path, "grid": grid_path}
    json.dump({k: scene[k] for k in ("seed", "dist", "pile_pick", "truck_dump", "truck_fp", "obstacles")},
              open(os.path.join(out, "scene.json"), "w"), indent=2)
    return scene


def build_gcrl_env():
    """GCRL 실행 env(obstacle_training=True, config.yaml 구성). 장애물은 apply_obstacles로 주입."""
    return render_scene.build_env()


def reset_with_obstacles(env, seed, obstacles):
    """env reset(HOME) 후 미션 장애물 주입(GT) → 관측 반환. (interface.apply_obstacles 재사용)"""
    env.reset(seed=seed)
    interface.apply_obstacles(env, obstacles)
    mujoco.mj_forward(env.model, env.data)
    return env._get_obs()


def rollout_mission(env, model, seed, obstacles, goals, sub_steps=150, rec=None,
                    tol_xy=STAGE_TOL_XY, adv_tol=REACH_TOL,
                    pile_pick=None, truck_dump=None, truck_fp=None):
    """pile→truck 미션 relay 실행. goals=서브골 시퀀스(B=VLM plan / A=[pile,truck]).
    성공=흙더미 도달(stage1) & 트럭 footprint 도달(stage2) & 충돌 0.
    pile_pick/truck_dump/truck_fp 주면 그 per-seed 타깃으로 채점(랜덤화). 없으면 모듈 고정값."""
    pile_pick = PILE_PICK if pile_pick is None else pile_pick
    truck_dump = TRUCK_DUMP if truck_dump is None else truck_dump
    fp = TRUCK_FOOTPRINT if truck_fp is None else truck_fp
    obs = reset_with_obstacles(env, seed, obstacles)
    pile, truck = np.asarray(pile_pick, float), np.asarray(truck_dump, float)
    waypoints = [np.asarray(g, float) for g in goals] if goals else [pile.copy(), truck.copy()]
    draw = waypoints if rec is not None else None
    collided, total, min_pile_xy, path = False, 0, np.inf, 0.0
    reached_truck = False                                    # tip이 트럭 footprint 안에 든 적 있으면 도달
    last_tip = env._tip().copy()
    if rec is not None:
        rec.reset()
    for wp in waypoints:
        env._desired = wp.copy()
        obs = env._get_obs()
        if rec is not None:
            rec.grab(env.data, wp, waypoints=draw)
        for _ in range(sub_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            total += 1
            tip = env._tip()
            path += float(np.linalg.norm(tip - last_tip)); last_tip = tip.copy()
            min_pile_xy = min(min_pile_xy, float(np.linalg.norm(tip[:2] - pile[:2])))
            if _in_truck_fp(tip[:2], fp=fp):
                reached_truck = True
            if info.get("min_gap", 1e9) <= 0.0:
                collided = True
            if rec is not None:
                rec.grab(env.data, wp, waypoints=draw)
            if float(np.linalg.norm(tip - wp)) < adv_tol:
                break
            if trunc:
                break
        if env._steps >= env.max_steps:
            break
    d_truck_xy = float(np.linalg.norm(env._tip()[:2] - truck[:2]))
    stage1 = bool(min_pile_xy < tol_xy)
    stage2 = bool(reached_truck)                             # 트럭 footprint 진입 여부

    success = stage1 and stage2
    return dict(stage1=int(stage1), stage2=int(stage2), success=int(success),
                collided=int(collided), safe=int(success and not collided),
                d_pile=round(float(min_pile_xy), 3), d_truck=round(d_truck_xy, 3),
                path=round(path, 2), steps=total, n_sub=len(goals))
