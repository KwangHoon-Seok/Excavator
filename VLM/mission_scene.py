"""
상차(loading) 미션 씬 — 지면 위 흙더미(pile) + 덤프트럭(truck) 고정 배치.
plane 지면(hfield 없음) 위에 얹으며, 장애물(랜덤 박스)은 GCRL env가 별도 주입.
task: "흙더미에서 덤프트럭으로 이동" — 둘 다 z>0(굴착기 z≥0 학습범위 내, 지면 구멍 불필요).

흙더미는 흙색 ellipsoid를 크기·위치 다양하게 군집 → 불규칙 둔덕(rounded mound)으로 사실감.
VLM(RGB)이 "둥근 흙더미"(vs 각진 회색 장애물 박스)로 구분 가능.

  build_model()  → 흙더미+트럭 주입 MjModel (plane 지면)
  PILE_PICK      → 흙더미 상단 픽업점(GCRL goal)   TRUCK_DUMP → 트럭 적재함 위 투하점
"""
import os
import numpy as np
import mujoco
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, "excavator.xml")
_CFG = yaml.safe_load(open(os.path.join(ROOT, "Model", "Version4", "config.yaml")))
_HOME_TIP_XY = np.array([0.0, 3.62])          # 시작자세 tip 근사(장애물 회피용)

# ── 고정 배치(튜닝) ──────────────────────────────────────────
TRUCK_OBJ = os.path.join(ROOT, "dump_truck",
                         "Vocational_Truck_v1_L1.123c070e9773-cb09-496b-b3ba-6bae17413534",
                         "16758_Vocational_Truck_v1_NEW.obj")
TRUCK_SCALE = 0.015
TRUCK_QUAT = "0.707 0 0 0.707"
TRUCK_XY = (3.2, 2.6)            # 오른쪽 가장자리(θ≈+59°)
PILE_XY = (-3.0, 1.5)           # 왼쪽 가장자리(θ≈-57°) 흙더미

PILE_PEAK_Z = 1.00
PILE_PICK = [PILE_XY[0], PILE_XY[1], PILE_PEAK_Z - 0.15]   # 흙더미 상단 픽업점(GCRL goal)
TRUCK_DUMP = [TRUCK_XY[0], TRUCK_XY[1], 1.6]               # 트럭 적재함 위 투하점(대략)


def _pile_clods(seed=42, n=52, base_r=0.4, peak_z=PILE_PEAK_Z):
    """각진 흙덩이(작은 박스) 군집을 원뿔 포락(넓은 밑동→좁은 봉우리, 안식각) 안에 랜덤 배치·회전.
    매끈한 ellipsoid 대신 불규칙 각덩이 → granular한 진짜 흙더미 질감. (결정론적 seed)"""
    rng = np.random.default_rng(seed)
    clods = []
    for _ in range(n):
        z = rng.uniform(0.0, peak_z * 0.90)
        rmax = base_r * (1.0 - z / peak_z)               # 높을수록 좁게(안식각)
        r = rmax * np.sqrt(rng.uniform(0.0, 1.0))        # 면적가중
        th = rng.uniform(0, 2 * np.pi)
        dx, dy = r * np.cos(th), r * np.sin(th)
        s = rng.uniform(0.10, 0.22)                      # 흙덩이 크기
        size = (s, s * rng.uniform(0.7, 1.0), s * rng.uniform(0.55, 0.85))
        u1, u2, u3 = rng.uniform(0, 1, 3)                # 균일 랜덤 쿼터니언(불규칙 회전)
        q = (np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
             np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3))
        sh = 0.42 + rng.uniform(-0.04, 0.06)             # 흙색 명암 변화
        rgba = f"{sh:.3f} {sh*0.73:.3f} {sh*0.43:.3f} 1"
        clods.append((dx, dy, z + s * 0.5, size, q, rgba))
    return clods


_PILE_CLODS = _pile_clods()


# 미션 장애물 높이 상한 — 팔이 "위로 통과" 가능한 높이로 제한(통과 불가 벽 방지). 학습(≤2.8)과 별개.
MISSION_HGT_RANGE = (0.6, 1.5)


def mission_obstacles(seed, k_range=(2, 4)):
    """평가용 랜덤 장애물 박스 — 흙더미↔트럭 traverse 중앙대에 배치.
    length/width는 학습 분포, height는 MISSION_HGT_RANGE(넘어갈 수 있게 상한) 사용.
    지면(z=0)에 붙고, 흙더미·트럭·시작자세·서로와 겹치지 않게. GCRL env엔 좌표만 넘김(interface.apply_obstacles).
    반환: [{"center":[x,y,z], "half":[hx,hy,hz]}, ...] (결정론적 seed)."""
    rng = np.random.default_rng(int(seed) + 10007)
    lr, wr, hr = _CFG["obs_len_range"], _CFG["obs_wid_range"], MISSION_HGT_RANGE
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    # (중심, 회피여유): 트럭은 mesh footprint가 커서(중심반경≈120·scale) 큰 여유 필요. 흙더미·시작자세는 작음.
    truck_clear = 120.0 * TRUCK_SCALE + 0.4          # scale 0.015 → ≈2.2m
    avoid = [(np.array(PILE_XY), 0.9), (np.array(TRUCK_XY), truck_clear), (_HOME_TIP_XY, 0.8)]
    obs = []
    for _ in range(k * 12):
        if len(obs) >= k:
            break
        half = np.array([rng.uniform(*lr), rng.uniform(*wr), rng.uniform(*hr)]) / 2.0
        c = np.array([rng.uniform(-1.6, 1.6), rng.uniform(1.5, 3.4), half[2]])   # 중앙 traverse대(트럭 x범위 앞은 제외)
        rad = float(max(half[0], half[1]))
        if any(np.linalg.norm(c[:2] - a) < rad + clr for a, clr in avoid):
            continue
        if any(np.linalg.norm(c[:2] - np.array(o["center"][:2])) < rad + max(o["half"][:2]) + 0.2 for o in obs):
            continue
        obs.append({"center": [round(float(v), 3) for v in c], "half": [round(float(v), 3) for v in half]})
    return obs


def _obstacles_xml(obs):
    """평가 렌더용 장애물 박스(시각 전용, 회색 — 흙더미 갈색·트럭 노랑과 구분)."""
    s = ""
    for i, o in enumerate(obs):
        c, h = o["center"], o["half"]
        s += (f'    <body name="mobs{i}" pos="{c[0]} {c[1]} {c[2]}">\n'
              f'      <geom name="mobs{i}_g" type="box" size="{h[0]} {h[1]} {h[2]}" '
              f'rgba="0.55 0.55 0.6 1" contype="0" conaffinity="0"/>\n    </body>\n')
    return s


def _asset_xml():
    return f'    <mesh name="dumptruck" file="{TRUCK_OBJ}" scale="{TRUCK_SCALE} {TRUCK_SCALE} {TRUCK_SCALE}"/>\n'


def _pile_xml(cx, cy):
    # 시각 전용(contype=0 conaffinity=0) — 픽업점은 좌표로 주므로 물리충돌 불필요
    s = f'    <body name="pile" pos="{cx} {cy} 0">\n'
    for i, (dx, dy, z, size, q, rgba) in enumerate(_PILE_CLODS):
        sx, sy, sz = size
        s += (f'      <geom name="pile{i}" type="box" pos="{dx:.3f} {dy:.3f} {z:.3f}" '
              f'quat="{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}" '
              f'size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{rgba}" contype="0" conaffinity="0"/>\n')
    return s + "    </body>\n"


def _truck_xml(cx, cy, quat=TRUCK_QUAT):
    return (f'    <body name="truck" pos="{cx} {cy} 0" quat="{quat}">\n'
            f'      <geom type="mesh" mesh="dumptruck" rgba="0.72 0.56 0.20 1" contype="0" conaffinity="0"/>\n'
            f'    </body>\n')


def build_model(seed=0):
    """흙더미+트럭(고정) + 랜덤 장애물(seed) 주입 MjModel + 장애물 spec 반환.
    반환: (model, obstacles). obstacles는 scene.json/interface.apply_obstacles로 GCRL에 전달."""
    obstacles = mission_obstacles(seed)
    xml = open(XML).read()
    xml = xml.replace("</asset>", _asset_xml() + "  </asset>")
    xml = xml.replace("</worldbody>", _pile_xml(*PILE_XY) + _truck_xml(*TRUCK_XY)
                      + _obstacles_xml(obstacles) + "  </worldbody>")
    return mujoco.MjModel.from_xml_string(xml), obstacles


def gridmap_bodies(obstacles):
    """gridmap object_bodies(높이 blob) — 흙더미·트럭·장애물 모두 포함(mesh도 정점으로 높이 검출).
    종류(source/target/avoid) 구분은 VLM이 RGB(외형)↔gridmap(위치) 대조로 판단."""
    return ["pile", "truck"] + [f"mobs{i}" for i in range(len(obstacles))]


if __name__ == "__main__":   # 렌더 확인
    from PIL import Image
    m, _obs = build_model(seed=0); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 1280)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cabin_cam")
    m.cam_fovy[cid] = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(135) / 2) / (1280 / 480)))
    r.update_scene(d, camera="cabin_cam")
    out = os.path.join(os.path.dirname(__file__), "out", "_mission_scene.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(r.render()).save(out)
    print("saved:", out, "| PILE_PICK", PILE_PICK, "TRUCK_DUMP", TRUCK_DUMP)
