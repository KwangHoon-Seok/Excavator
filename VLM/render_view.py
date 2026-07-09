"""
cabin_cam(굴착기 cabin 상단-우측 RGB 카메라) 렌더 → VLM 입력 이미지.

  conda run -n seok python VLM/render_view.py            # 데모(rock 주입) 와이드뷰
  conda run -n seok python VLM/render_view.py --plain    # 현재 xml 그대로
  conda run -n seok python VLM/render_view.py --w 1280 --h 480 --swing 0.4

수평 커버(-90~90 목표)는 종횡비로 결정: 넓을수록↑. 순수 핀홀은 완전 180° 불가(왜곡).
산출: VLM/out/cabin_view.png
"""
import os, sys, argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import terrain

XML = os.path.join(ROOT, "excavator.xml")
OUT = os.path.join(HERE, "out")

DEMO_ROCKS = """
    <body name="rock1" pos="0.6 2.6 -0.245"><freejoint/>
      <geom class="terrain" type="box" size="0.22 0.22 0.2" rgba="0.45 0.38 0.30 1"/></body>
    <body name="rock2" pos="-0.9 2.0 -0.295"><freejoint/>
      <geom class="terrain" type="sphere" size="0.15" rgba="0.5 0.42 0.34 1"/></body>
    <body name="rock3" pos="1.4 3.4 -0.28"><freejoint/>
      <geom class="terrain" type="box" size="0.18 0.18 0.16" rgba="0.5 0.44 0.36 1"/></body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--w", type=int, default=1280, help="렌더 폭")
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fovx", type=float, default=135.0, help="목표 수평 FOV(도) — 해상도에 맞춰 fovy 자동설정")
    ap.add_argument("--swing", type=float, default=0.0, help="선회각(rad) — 카메라도 함께 회전")
    ap.add_argument("--cam", default="cabin_cam")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    xml = open(XML).read()
    if not a.plain:
        xml = xml.replace("</worldbody>", DEMO_ROCKS + "  </worldbody>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    terrain.reset_terrain(m)
    d.qpos[0] = a.swing                       # swing → 카메라(cabin) 회전
    mujoco.mj_forward(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, a.cam)
    if cid < 0:
        raise SystemExit(f"camera '{a.cam}' not found")
    # 목표 수평 FOV(fovx)에 맞춰 fovy를 해상도별로 자동설정 → 해상도 바꿔도 커버 일정
    aspect = a.w / a.h
    fovy = 2 * np.arctan(np.tan(np.deg2rad(a.fovx) / 2) / aspect)
    m.cam_fovy[cid] = np.rad2deg(fovy)
    print(f"cam '{a.cam}' 수평 FOV={a.fovx:.0f}°  aspect={aspect:.2f}  → fovy={m.cam_fovy[cid]:.2f}°")

    r = mujoco.Renderer(m, a.h, a.w)
    r.update_scene(d, camera=a.cam)
    img = r.render()
    r.close()

    from PIL import Image
    p = os.path.join(OUT, "cabin_view.png")
    Image.fromarray(img).save(p)
    print("saved:", p)


if __name__ == "__main__":
    main()
