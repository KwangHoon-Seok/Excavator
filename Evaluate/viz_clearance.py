"""
논문 figure: 링크별 최근접 장애물 clearance (실제 직사각형 박스 환경).
각 링크(캡슐)에서 '자기 최근접 박스' 표면까지 화살표 + 거리. MuJoCo 렌더 + matplotlib 저장.
"""
import os, json
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, "excavator.xml")
PRIMS = os.path.join(ROOT, "Model", "arm_primitives.json")
OUT = os.path.join(ROOT, "Evaluate", "plots")

LINK_COL = {"boom": (0.20, 0.55, 0.95), "stick": (0.95, 0.58, 0.12), "bucket": (0.86, 0.20, 0.42)}
LINK_LABEL = {"boom": "Boom", "stick": "Arm (stick)", "bucket": "Bucket"}
POSE = [0.0, -0.66, -1.83, 0.30]    # [swing, boom, stick, bucket] rad — boom 최대 + stick 최대접힘 = 팔 최대로 듦
# 장애물 1개 (팔의 오른쪽 +x): name -> (center xyz, half xyz)
MANUAL_BOXES = {
    "obs": (np.array([0.95, 2.35, 0.70]), np.array([0.28, 0.28, 0.70])),
}


def seg_box_closest(A, B, c, h, n=200):
    best = (None, None, 1e9)
    for t in np.linspace(0, 1, n):
        p = A + t * (B - A)
        q = np.clip(p, c - h, c + h)
        d = np.linalg.norm(p - q)
        if d < best[2]:
            best = (p.copy(), q.copy(), float(d))
    return best


def build_caps(model, data):
    prims = json.load(open(PRIMS))
    bl = {"part02_cmpl": "boom", "part03": "stick"}
    caps = {"boom": [], "stick": [], "bucket": []}
    for pr in prims:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pr["body"])
        Rm = data.xmat[bid].reshape(3, 3); p = data.xpos[bid]
        caps[bl[pr["body"]]].append((p + Rm @ np.array(pr["A"]),
                                     p + Rm @ np.array(pr["B"]), pr["R"]))
    p4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "part04")
    ta = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "bucket_tip_pos")]
    tip = data.sensordata[ta:ta + 3].copy()
    o = data.xpos[p4].copy()                         # 손목~중간까지만(버킷 셸, 지면 위 유지)
    caps["bucket"].append((o, o + 0.5 * (tip - o), 0.20))
    return caps


def place_boxes(caps):
    """링크별 대표 캡슐 옆에 박스 하나씩 — 목표 clearance≈0.30 m가 되도록 perp 오프셋 이분탐색."""
    sides = {"boom": +1, "stick": -1, "bucket": +1}
    boxes = {}
    for lnm, cl in caps.items():
        A, B, R = max(cl, key=lambda s: np.hypot(*(0.5 * (s[0] + s[1])[:2])))  # 가장 바깥 캡슐
        mid = 0.5 * (A + B); dirv = (B - A) / (np.linalg.norm(B - A) + 1e-9)
        perp = np.array([-dirv[1], dirv[0], 0.0]); perp /= (np.linalg.norm(perp) + 1e-9)
        hz = max(0.5, mid[2] + 0.15); half = np.array([0.24, 0.24, hz])
        lo, hi = 0.3, 1.5
        for _ in range(24):
            L = 0.5 * (lo + hi)
            c = mid + perp * sides[lnm] * L; c[2] = hz
            gap = seg_box_closest(A, B, c, half)[2] - R
            if gap < 0.18: lo = L
            else: hi = L
        c = mid + perp * sides[lnm] * L; c[2] = hz
        boxes[lnm] = (c, half)
    return boxes


def inject_boxes(xml_path, boxes):
    xml = open(xml_path).read(); b = ""
    for nm, (c, h) in boxes.items():
        b += (f'    <body name="box_{nm}" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}">\n'
              f'      <geom type="box" size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}" '
              f'rgba="0.56 0.56 0.60 1" contype="0" conaffinity="0"/>\n    </body>\n')
    return xml.replace("</worldbody>", b + "  </worldbody>")


def main():
    m0 = mujoco.MjModel.from_xml_path(XML); d0 = mujoco.MjData(m0)
    d0.qpos[:4] = POSE; mujoco.mj_forward(m0, d0)
    caps0 = build_caps(m0, d0)
    boxes = MANUAL_BOXES                              # 수동 박스 배치

    model = mujoco.MjModel.from_xml_string(inject_boxes(XML, boxes))
    data = mujoco.MjData(model); data.qpos[:4] = POSE; mujoco.mj_forward(model, data)
    caps = build_caps(model, data)
    box_list = [(nm, c, h) for nm, (c, h) in boxes.items()]

    # 링크별 '전역' 최근접 박스 + witness 점
    results = {}
    for lnm, cl in caps.items():
        best = (None, None, 1e9, None)
        for (A, B, R) in cl:
            for bnm, c, h in box_list:
                ps, pb, dseg = seg_box_closest(A, B, c, h)
                gap = dseg - R
                if gap < best[2]:
                    dirv = (pb - ps) / (dseg + 1e-9)
                    best = (ps + R * dirv, pb, gap, bnm)
        results[lnm] = best
        print(f"{lnm:7s} → box_{best[3]}  clearance={best[2]*100:5.1f} cm")

    # ── 렌더 ──
    H, W = 900, 1200
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.35, 2.35, 0.65]; cam.distance = 7.6
    cam.azimuth = 112; cam.elevation = -18
    r.update_scene(data, cam, mujoco.MjvOption())
    scn = r.scene

    def _g(): return scn.geoms[scn.ngeom]
    def cap(a, b, rad, rgba):
        mujoco.mjv_initGeom(_g(), mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(_g(), mujoco.mjtGeom.mjGEOM_CAPSULE, rad, a, b); scn.ngeom += 1
    def arrow(a, b, rgba, w=0.03):
        mujoco.mjv_initGeom(_g(), mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(_g(), mujoco.mjtGeom.mjGEOM_ARROW, w, a, b); scn.ngeom += 1
    def sph(p, rad, rgba):
        mujoco.mjv_initGeom(_g(), mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, np.float32)); scn.ngeom += 1

    for lnm, cl in caps.items():
        col = LINK_COL[lnm]
        for (A, B, R) in cl:
            cap(A, B, R, (*col, 0.28))
        p_surf, p_box, gap, bnm = results[lnm]
        # 연결선: 표면↔표면 전체를 캡슐로 확실히 span + 양 끝 witness 구
        cap(p_surf, p_box, 0.016, (*col, 1.0))
        sph(p_surf, 0.05, (*col, 1.0)); sph(p_box, 0.05, (*col, 1.0))
        # 화살촉: 박스쪽 끝 짧게(선이 실제로 표면에 닿음을 표시)
        u = (p_box - p_surf) / (np.linalg.norm(p_box - p_surf) + 1e-9)
        arrow(p_box - u * 0.14, p_box, (*col, 1.0), w=0.05)

    img = r.render(); r.close()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.2, 6.9))
    ax.imshow(img); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("png", "pdf"):
        fp = os.path.join(OUT, f"clearance_boxes.{ext}")
        fig.savefig(fp, bbox_inches="tight", pad_inches=0, dpi=300); print("saved:", fp)
    plt.close(fig)


if __name__ == "__main__":
    main()
