"""
학습 환경(obstacle-aware GCRL) 세팅 검증 — MuJoCo 뷰어 + 콘솔 리포트.
매 reset마다 다음 3가지를 확인(행동/롤아웃 없음, 홈 자세 스냅샷):
  ① 장애물 생성      - 개수·크기(LxWxH)·위치가 작업공간 안에 적절히 배치되는지
  ② TTC 계산         - "최근접 링크가 장애물로 접근하는" qvel을 주입해 링크별 gap/TTC/severity 산출
  ③ 회피 보상        - ②의 ttc_severity가 reward_decomposed의 TTC 페널티로 실제 반영되는지(전체 보상 분해)

시작 자세는 배치 제약(#1)으로 장애물과 여유가 있어 정지 severity=0 → 접근 qvel을 주입해야 TTC/보상이 유의미.
오버레이: boom(초록)/stick(파랑) 캡슐, 장애물 박스(빨강)+내접구(노랑), goal(초록), 최근접 clearance(심각도색).
매 dwell초 새 씬으로 순환. 실행:
  conda run -n seok python Model/validation_train_env.py [--dwell 5] [--seeds 3,7] [--goal 0,2,3] [--approach 0.6]
"""
import os, sys, time, argparse
import numpy as np
import yaml
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from excavator_env import ExcavatorRHEREnv
import reward as R


def rot_to(zt):
    z = zt / np.linalg.norm(zt); a = np.array([0, 0, 1.])
    v = np.cross(a, z); s = np.linalg.norm(v); c = a @ z
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1., -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def sev_color(sev):
    if sev > 0.7:  return (0.92, 0.12, 0.12)
    if sev > 0.4:  return (1.0, 0.55, 0.0)
    if sev > 0.15: return (0.92, 0.86, 0.0)
    return (0.24, 0.82, 0.24)


def add_sphere(scn, c, r, rgba):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([r, 0, 0.]),
                        np.ascontiguousarray(c, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_cap(scn, A, B, R_, rgba):
    if scn.ngeom >= scn.maxgeom: return
    L = np.linalg.norm(B - A)
    if L < 1e-6: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.array([R_, L / 2, 0.]),
                        np.ascontiguousarray((A + B) / 2, float),
                        np.ascontiguousarray(rot_to(B - A).flatten(), float), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def set_approach_qvel(e, target_speed):
    """최근접(min-gap) 링크의 닫힌점이 장애물 쪽(법선 n)으로 target_speed[m/s]만큼 접근하도록 qvel 설정.
    → 그 링크의 TTC=gap/target이 유의미해지고, 보상 TTC 페널티가 활성. (실제 학습 중 접근 순간을 재현)"""
    m, d = e.model, e.data
    d.qvel[:] = 0.0
    best = None
    for p1, n, gap, body, lnm in e._prim_pairs():
        if gap > 0 and (best is None or gap < best[2]):
            best = (p1, n, gap, body, lnm)
    if best is None:
        return None
    p1, n, gap, body, lnm = best
    J = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, J, None, np.ascontiguousarray(p1, float), body)
    g4 = J[:, :4].T @ n                                   # closing speed의 관절속도 gradient
    den = float(g4 @ g4)
    if den < 1e-9:
        return None
    qv = (target_speed / den) * g4                       # closing = qv·g4 = target
    vmax = e.dmax / (e.frame_skip * m.opt.timestep)      # 학습 스텝당 최대 관절속도(rad/s)로 캡
    d.qvel[:4] = np.clip(qv, -vmax, vmax)
    mujoco.mj_forward(m, d)
    return lnm


def link_ttc(e):
    info = {}
    for p1, n, gap, body, lnm in e._prim_pairs():
        cl = float(e._pvel(p1, body) @ n)
        ttc = (gap / cl) if (cl > 1e-3 and gap > 0) else (0.0 if gap <= 0 else 99.0)
        if lnm not in info or gap < info[lnm][0]:
            info[lnm] = (gap, ttc, cl, p1.copy(), p1 + n * max(gap, 0.0))
    return {k: (g, t, cl, (1.0 if g <= 0 else float(np.exp(-t / e.ttc_max))), p1, op)
            for k, (g, t, cl, p1, op) in info.items()}


def draw_overlay(e, viewer, ttc):
    m, d = e.model, e.data
    scn = viewer.user_scn; scn.ngeom = 0
    for bid, la, lb, R_, lnm in e._caps:
        Rm = d.xmat[bid].reshape(3, 3); p = d.xpos[bid]
        add_cap(scn, p + Rm @ la, p + Rm @ lb, R_, (0.2, 0.9, 0.2, 0.45) if lnm == 'boom' else (0.2, 0.5, 1.0, 0.45))
    for c, r in e._obs_sph:
        add_sphere(scn, c, r, (1.0, 0.9, 0.1, 0.28))
    add_sphere(scn, e._desired, 0.16, (0.1, 0.95, 0.1, 0.9))
    for lnm, (gap, tv, cl, sev, p1, op) in ttc.items():
        col = sev_color(sev)
        if 0 < gap < 2.5:
            add_cap(scn, p1, op, 0.03, col + (1.0,))
        add_sphere(scn, p1, 0.06, col + (1.0,))


def report(e, seed, appr_link, ttc):
    """콘솔에 3가지 검증 리포트."""
    m, d = e.model, e.data
    g = e._desired
    print('\n══ seed %d | goal(%.2f,%.2f,%.2f) ══' % (seed, g[0], g[1], g[2]))
    # ① 장애물 생성
    print(' ① 장애물 %d개:' % len(e._active_obs))
    for og in e._active_obs:
        c = m.body_pos[m.geom_bodyid[og]]; hw = 2 * m.geom_size[og]
        print('    pos(%.2f,%.2f) r=%.2f  size LxWxH=(%.2f,%.2f,%.2f) 바닥z=%.3f'
              % (c[0], c[1], np.hypot(c[0], c[1]), hw[0], hw[1], hw[2], c[2] - m.geom_size[og][2]))
    # ② TTC (접근 가정)
    if not e._active_obs:
        print(' ② TTC: 장애물 없음 → 이 씬은 순수 도달(회피 대상 없음)')
    else:
        print(' ② TTC (최근접 링크 "%s"가 장애물로 접근한다고 가정):' % (appr_link or '없음'))
        for lnm in ['boom', 'stick', 'bucket']:
            if lnm not in ttc:
                print('    %-7s (원거리 — 페어 없음)' % lnm); continue
            gap, tv, cl, sev, _, _ = ttc[lnm]
            lab = 'HIT' if gap <= 0 else ('TTC>99' if tv > 90 else 'TTC=%.2fs' % tv)
            print('    %-7s gap=%+.2f  closing=%+.2fm/s  %-9s sev=%.2f' % (lnm, gap, cl, lab, sev))
    # ③ 보상 (ttc_severity → reward 페널티)
    sev = e._ttc_severity()
    ag = e._tip() / e.pos_scale; dg = e._desired / e.pos_scale
    info = {"ttc_severity": sev, "rate_pen": 0.0}
    r_total = float(e.reward_fn(ag, dg, info, e.eps))
    swing, reach, succ = R.decomposed_terms(ag, dg, e.eps)
    ttc_pen = e.w_ttc * sev
    print(' ③ 회피 보상: ttc_severity=%.3f → TTC penalty = -w_ttc(%.2f)×%.3f = %+.3f'
          % (sev, e.w_ttc, sev, -ttc_pen))
    print('    전체 reward_decomposed=%+.3f  (sparse=%+d, swing=%+.3f, reach=%+.3f, TTC=%+.3f)'
          % (r_total, 0 if succ else -1, -0.5 * float(swing), -0.5 * float(reach), -ttc_pen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, 'config.yaml'))
    ap.add_argument('--seeds', default=None, help='고정 시드들(쉼표구분, 순환). 미지정 시 매 dwell 새 시드')
    ap.add_argument('--goal', default=None, help='고정 goal "x,y,z". 미지정 시 envelope 랜덤')
    ap.add_argument('--dwell', type=float, default=5.0, help='씬 자동 전환 간격(초). 0=고정')
    ap.add_argument('--approach', type=float, default=0.6, help='TTC 검증용 가정 접근속도(m/s)')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else None
    goal = np.array([float(x) for x in a.goal.split(',')]) if a.goal else None

    e = ExcavatorRHEREnv(reward=cfg.get('reward', 'reward_decomposed'),
                         random_start=cfg.get('random_start', False),
                         normalize_goal=cfg.get('normalize_goal', True),
                         goal_azimuth_limit_deg=cfg.get('goal_azimuth_limit_deg', None),
                         eps=cfg.get('eps', 0.2), max_steps=cfg.get('max_steps', 400),
                         frame_skip=cfg.get('frame_skip', 20),
                         obstacle_training=True,
                         n_obstacles=cfg.get('n_obstacles', 3),
                         obs_len_range=cfg.get('obs_len_range', (0.3, 0.9)),
                         obs_wid_range=cfg.get('obs_wid_range', (0.3, 0.9)),
                         obs_hgt_range=cfg.get('obs_hgt_range', (0.3, 0.9)),
                         obs_start_margin=cfg.get('obs_start_margin', 0.30),
                         obs_goal_margin=cfg.get('obs_goal_margin', 0.40),
                         obs_path_relevant=cfg.get('obs_path_relevant', 1),
                         obs_path_corridor=cfg.get('obs_path_corridor', 0.6),
                         obs_place_tries=cfg.get('obs_place_tries', 20),
                         ttc_max=cfg.get('ttc_max', 6.0), w_ttc=cfg.get('w_ttc', 0.3))
    m, d = e.model, e.data
    m.geom_rgba[list(e._obs_geoms)] = [0.9, 0.1, 0.1, 0.35]
    opts = {"goal": np.asarray(goal, float)} if goal is not None else None

    def seed_at(i):
        return seeds[i % len(seeds)] if seeds else i

    def load(i):
        seed = seed_at(i)
        e.reset(seed=seed, options=opts)
        appr = set_approach_qvel(e, a.approach)      # 최근접 링크 접근 qvel 주입 → TTC/보상 유의미
        ttc = link_ttc(e)
        return seed, appr, ttc

    print('MuJoCo 뷰어 — 매 %.0fs 새 씬. 콘솔에 장애물/TTC/보상 리포트 출력. 창 닫으면 종료.' % a.dwell)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0, 1.0, 0.9]; viewer.cam.distance = 11.5
        viewer.cam.azimuth = 140; viewer.cam.elevation = -12
        i = 0
        seed, appr, ttc = load(i); draw_overlay(e, viewer, ttc); report(e, seed, appr, ttc)
        viewer.sync(); last = time.time()
        while viewer.is_running():
            if a.dwell > 0 and time.time() - last > a.dwell:
                i += 1
                seed, appr, ttc = load(i); draw_overlay(e, viewer, ttc); report(e, seed, appr, ttc)
                viewer.sync(); last = time.time()
            else:
                viewer.sync()
            time.sleep(0.02)


if __name__ == '__main__':
    main()
