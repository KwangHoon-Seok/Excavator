"""
ExcavatorRHEREnv — 굴착기 waypoint reaching 환경 (SAC + HER, Gymnasium).

task: bucket_tip을 자유공간의 목표 waypoint(x,y,z)로 보낸다(단일 goal-conditioned).
- obs: observation=[qpos4, qvel4] (정규화), achieved_goal=[tip xyz], desired_goal=[waypoint xyz].
- reward: 거리 기준(reward.dense). HER 호환(ag,dg,info만으로 계산).
- 액션 [-1,1]^4 → 증분 역정규화 tgt=clip(qpos+a·Δmax). MuJoCo ctrlrange는 물리(rad).
- 목표는 매 reset마다 _sample_reachable_waypoint()로 랜덤 생성(또는 options={"goal":[x,y,z]}로 지정).
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import reward as R  # Model/reward.py  # noqa: E402

XML = os.path.join(ROOT, "excavator.xml")
HOME = np.array([0.0, -0.35, -0.7, 0.0])       # 시작(팔 전방 뻗음) 자세 → tip≈(0.0, 3.62, 1.01)
Z_MIN, R_MIN = 0.0, 1.0                    # waypoint 도달영역 필터(지면=0 위 / 몸체 밖). 굴착 goal은 배포 때 options로 직접 지정


class ExcavatorRHEREnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, frame_skip=20, max_steps=300, eps=0.3, reward="sparse",
                 random_start=False, normalize_goal=True, goal_azimuth_limit_deg=None,
                 obstacle_training=False, n_obstacles=3,
                 obs_len_range=(0.3, 0.9), obs_wid_range=(0.3, 0.9), obs_hgt_range=(0.3, 0.9),
                 obs_start_margin=0.30, obs_goal_margin=0.40,
                 obs_path_relevant=1, obs_path_corridor=0.6, obs_place_tries=20,
                 ttc_max=6.0, w_ttc=0.3, render_mode=None):
        super().__init__()
        # obstacle_training: True면 장애물 회피 GCRL(obs에 링크별 clearance, 보상에 TTC severity)
        self.obstacle_training = obstacle_training
        self.n_obstacles = n_obstacles     # 에피소드당 활성 장애물 수 상한(1~이 값 랜덤)
        # 장애물 박스 전체 치수(length=x, width=y, height=z) 랜덤 범위(m). reset마다 각 축 독립 샘플
        self.obs_len_range = tuple(obs_len_range)   # 길이(x)
        self.obs_wid_range = tuple(obs_wid_range)   # 폭(y)
        self.obs_hgt_range = tuple(obs_hgt_range)   # 높이(z)
        # 배치 제약(rejection sampling): 시작자세 팔/ goal과 최소여유, 경로 관련성 필터
        self.obs_start_margin = obs_start_margin    # #1 시작자세 팔과 최소 gap(m)
        self.obs_goal_margin = obs_goal_margin      # #2 goal과 최소 gap(m) ≈ eps+α
        self.obs_path_relevant = obs_path_relevant  # #3 start_tip→goal 경로 corridor 내에 강제할 장애물 수
        self.obs_path_corridor = obs_path_corridor  # #3 경로선분까지 수평거리 허용폭(m)
        self.obs_place_tries = obs_place_tries      # 후보 재시도 상한(실패 시 그 장애물 비활성)
        self.ttc_max = ttc_max             # TTC severity 지수커널 시간상수 τ(s): severity=exp(-TTC/τ)
        self.w_ttc = w_ttc                 # 보상 TTC 페널티 가중
        self._n_obs_geom = 4               # 주입할 장애물 geom 수(활성 ≤ 이 값)
        if obstacle_training:              # 장애물 box 주입(contype=4=물리 무시, mj_geomDistance만). 크기는 reset마다 랜덤
            xml = open(XML).read()
            # from_xml_string은 상대 meshdir을 CWD 기준으로 찾으므로, 실행 위치와 무관하도록 절대경로로 치환
            xml = xml.replace('meshdir="excavatorURDF"', 'meshdir="%s"' % os.path.join(ROOT, "excavatorURDF"))
            inj = ''.join('<body name="obs%d" pos="0 0 %d"><geom name="obs%d_g" type="box" '
                          'size="0.3 0.3 0.3" rgba="0.65 0.4 0.4 0.75" contype="4" conaffinity="0"/></body>'
                          % (i, 50 + i, i) for i in range(self._n_obs_geom))
            self.model = mujoco.MjModel.from_xml_string(xml.replace('</worldbody>', inj + '</worldbody>'))
        else:
            self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.eps = eps
        self.reward_fn = getattr(R, reward)   # reward.py의 함수 선택(sparse/dense/...)
        self.random_start = random_start      # True면 매 reset 관절 랜덤 시작자세(stitching·일반화)
        # goal 방위각을 시작 전방(카메라 방향) ±limit로 제한(카메라는 후면 객체 안 봄). None=전체 360°
        self.goal_az_limit = np.radians(goal_azimuth_limit_deg) if goal_azimuth_limit_deg else None
        self.render_mode = render_mode
        self._renderer = None

        m = self.model
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        self.ctrl_half = 0.5 * (self.ctrl_hi - self.ctrl_lo)
        self.dmax = np.array([0.12, 0.03, 0.05, 0.08])   # 스텝당 관절 최대 증분(rad) -> 최대증분 논리적 근거
        self.tip_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "bucket_tip_pos")]
        # 자기충돌 검사용 geom: 버킷 충돌박스(contype=2) vs cabin+boom 메시(contype=0이라 접촉엔진은 무시 → mj_geomDistance로 직접 계산)
        self._col_bucket = [g for g in range(m.ngeom) if m.geom_contype[g] == 2]
        _cab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "compact_excavator_turret_cabin_roller")
        _boom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part02_cmpl")
        self._col_body = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (_cab, _boom)]
        if self.obstacle_training:
            self._setup_obstacle_primitives()   # 팔 캡슐(body-local) + bkt박스 + 장애물 geom 등록

        self._desired = np.zeros(3)
        self._steps = 0
        self._build_reach_cells()        # 도달 envelope + home tip/거리 기준
        self.curr_r = None               # 커리큘럼 home-거리 한도(None=전체). set_curriculum로 조절

        # ── 정규화 스케일 ──
        dt = self.frame_skip * self.model.opt.timestep   # 제어주기(=20×0.002=0.04s)
        self.v_max = self.dmax / dt          # qvel은 항상 관절별 정규화(goal 토글과 무관)
        # goal/tip 정규화: normalize_goal이면 측정 작업반경, 아니면 1.0(=원본 유지)
        self.normalize_goal = normalize_goal
        self.pos_scale = self._reach_rmax if normalize_goal else 1.0
        self.eps_n = self.eps / self.pos_scale   # 성공 임계(정규화면 ~0.066, 원본이면 0.3)
        # reward_decomposed reach항 정규화 = pos_scale 단위 평면오차(r_err²+z_err²) 최대(측정값)
        r_err_max = (self._reach_rmax - R_MIN) / self.pos_scale
        z_err_max = (self._z_edges[-1] - Z_MIN) / self.pos_scale
        self.reach_norm = float(r_err_max ** 2 + z_err_max ** 2)
        R.REACH_NORM = self.reach_norm       # reward 모듈 주입(정규화면 ≈1.0, 원본이면 ≈20)
        R.W_TTC = self.w_ttc                 # 장애물 TTC 페널티 가중 주입

        # obstacle_training이면 링크별 clearance(n̂3+gap1)×3=12 추가 → 8→20
        obs_dim = 20 if self.obstacle_training else 8
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),  # qpos4+qvel4(+clearance12)
            achieved_goal=spaces.Box(-np.inf, np.inf, (3,), np.float32),  # tip xyz
            desired_goal=spaces.Box(-np.inf, np.inf, (3,), np.float32),   # waypoint xyz
        ))
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # ── helpers ──
    def _tip(self):
        return self.data.sensordata[self.tip_adr:self.tip_adr + 3].copy()

    def _self_collision(self, margin):
        """버킷 충돌박스와 cabin/boom 사이 최소거리 < margin이면 자기충돌(관통 포함)로 판정.
        cabin+arm은 swing축에서 통째로 회전 → 자기충돌 형상은 방위각 무관(→ (r,z) 마스크 전 θ 유효)."""
        ft = np.zeros(6)
        for a in self._col_bucket:
            for b in self._col_body:
                if mujoco.mj_geomDistance(self.model, self.data, a, b, margin, ft) < margin:
                    return True
        return False

    # ── 장애물 회피(obstacle_training) ──
    def _setup_obstacle_primitives(self):
        """팔 충돌 프리미티브(캡슐 body-local + bkt박스) + 장애물 geom 등록."""
        import json
        m = self.model
        self._obs_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obs%d_g" % i) for i in range(self._n_obs_geom)]
        self._obs_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obs%d" % i) for i in range(self._n_obs_geom)]
        prims = json.load(open(os.path.join(ROOT, "Model", "arm_primitives.json")))
        self._caps = []   # (body_id, A_local, B_local, R, link_name)
        for p in prims:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p["body"])
            lnm = "boom" if p["body"] == "part02_cmpl" else "stick"
            self._caps.append((bid, np.array(p["A"]), np.array(p["B"]), float(p["R"]), lnm))
        self._bkt_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] == 2]   # 버킷 4박스
        self._part04 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part04")
        self._ft = np.zeros(6)
        self._obs_sph = []      # 활성 장애물 박스의 구 근사 [(center_world, R), ...] (캡슐 거리용)
        self._active_obs = []   # 활성 장애물 box geom id (버킷 mj_geomDistance용)

    @staticmethod
    def _segcl(A, B, P):   # 선분 AB 위 P 최근접점
        AB = B - A; t = np.clip(np.dot(P - A, AB) / (np.dot(AB, AB) + 1e-9), 0, 1); return A + t * AB

    def _pvel(self, p, body):   # 강체 body 위 점 p의 속도 = J(p)·qvel (Jacobian, 회전 포함)
        jp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, None, np.ascontiguousarray(p, float), body)
        return jp @ self.data.qvel

    @staticmethod
    def _box_spheres(center, half):
        """축정렬 박스(center, half-extents) → 내접 구 세트(박스 length/width/height를 넘지 않음).
        R = 가장 짧은 half-extent → 세 축 모두 박스 안(내접). 긴 두 축을 따라 격자로 배치(구끼리 중첩 허용).
        끝면에 접하되 밖으로 안 나가도록 중심을 [-(h-R), h-R]에 둠. (사용자 요구: 실제 크기를 벗어나지 않는 근사)"""
        half = np.asarray(half, float)
        order = np.argsort(half)[::-1]                        # [긴축, 중간축, 짧은축]
        R = float(max(half[order[2]], 1e-3))                  # 최단 half = 내접 반경(어느 면도 안 넘음)

        def coords(h):
            span = h - R                                      # 중심 허용범위(끝면 밖으로 안 나감)
            if span <= 1e-6:
                return np.array([0.0])
            n = max(2, int(np.ceil(span / R)) + 1)            # spacing ≤ R → 충분한 중첩
            return np.linspace(-span, span, n)

        la, ma = np.eye(3)[order[0]], np.eye(3)[order[1]]     # 긴 두 축(월드 축정렬)
        return [center + la * t + ma * u
                for t in coords(half[order[0]]) for u in coords(half[order[1]])], R

    def _arm_prims_world(self):
        """시작자세 팔 프리미티브(월드): 캡슐 4개[(A,B,Rcap)] + 버킷박스 근사구[(center,R)].
        후보 장애물과의 최소여유(#1) 계산용. 현재 data(시작자세)에서 1회 캐시."""
        d = self.data
        caps = []
        for bid, la, lb, R, _lnm in self._caps:
            Rm = d.xmat[bid].reshape(3, 3); p = d.xpos[bid]
            caps.append((p + Rm @ la, p + Rm @ lb, R))
        pts = [(d.geom_xpos[g].copy(), float(self.model.geom_rbound[g])) for g in self._bkt_geoms]
        return caps, pts

    def _min_gap_to_arm(self, arm, spheres, R_sph, center, half):
        """후보 박스와 시작자세 팔 사이 최소 gap(실제 gap의 하한).
        - 캡슐(boom/stick): 박스 내접구(spheres,R_sph) 기준 — _prim_pairs와 동일식(정확 일치)
        - 버킷: 박스 전체(exact point-to-box) vs 버킷 bounding sphere(보수적) — 모서리 누수 방지"""
        caps, pts = arm
        mg = np.inf
        for cc in spheres:
            for A, B, Rcap in caps:
                p = self._segcl(A, B, cc)
                mg = min(mg, float(np.linalg.norm(cc - p)) - Rcap - R_sph)
        for pc, pr in pts:                                    # 버킷: 박스 exact - bounding sphere
            mg = min(mg, self._point_box_gap(pc, center, half) - pr)
        return mg

    @staticmethod
    def _point_box_gap(pt, center, half):
        """점 pt와 축정렬 박스(center, half) 사이 부호거리(내부=음수). goal 여유(#2)용."""
        q = np.abs(np.asarray(pt) - center) - half
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(np.max(q), 0.0))

    @staticmethod
    def _seg_point_dist_xy(a, b, p):
        """수평면(x,y)에서 점 p와 선분 a-b 사이 거리. 경로 관련성(#3)용."""
        a2, b2, p2 = a[:2], b[:2], p[:2]
        ab = b2 - a2; L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-9 else float(np.clip((p2 - a2) @ ab / L2, 0.0, 1.0))
        return float(np.linalg.norm(p2 - (a2 + t * ab)))

    def _obs_center_xy(self, start_tip, goal, require_path):
        """장애물 수평중심(x,y) 샘플. require_path면 start_tip→goal 경로 밴드 안에서 생성
        (경로 위 임의점 t + 측면 지터 ±corridor → 밴드 안 랜덤). 아니면 도달영역 랜덤 셀."""
        if require_path:
            t = self.np_random.uniform(0.25, 0.75)                 # 경로 상 위치(양끝 회피)
            base = start_tip[:2] + t * (goal[:2] - start_tip[:2])
            dxy = goal[:2] - start_tip[:2]; nrm = float(np.linalg.norm(dxy))
            perp = np.array([-dxy[1], dxy[0]]) / (nrm + 1e-9)      # 경로 수직(수평)
            off = self.np_random.uniform(-self.obs_path_corridor, self.obs_path_corridor)
            return base + perp * off
        lim = self.goal_az_limit
        # 작업공간 x-y와 일치: goal과 동일한 면적가중 반경(√uniform(r_min²,r_max²)) → 도달영역 환형 위 균일
        r = float(np.sqrt(self.np_random.uniform(self._reach_rmin ** 2, self._reach_rmax ** 2)))
        # 방위도 goal과 동일 기준: 시작 전방(fwd) ±lim (절대각이 아니라 전방기준 → 후면 장애물 방지)
        fwd = np.arctan2(start_tip[1], start_tip[0])
        th = (fwd + self.np_random.uniform(-lim, lim)) if lim is not None else self.np_random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(th), r * np.sin(th)])

    def _sample_one_obstacle(self, arm, start_tip, goal, require_path):
        """제약(#1 시작팔여유·#2 goal여유·#3 경로관련성)을 만족하는 장애물 1개 샘플.
        성공 시 (center, half, spheres, R), 실패(재시도 초과) 시 None."""
        for _ in range(self.obs_place_tries):
            dims = np.array([self.np_random.uniform(*self.obs_len_range),
                             self.np_random.uniform(*self.obs_wid_range),
                             self.np_random.uniform(*self.obs_hgt_range)])
            half = dims / 2.0
            xy = self._obs_center_xy(start_tip, goal, require_path)
            c = np.array([xy[0], xy[1], half[2]])          # 바닥이 지면(z=0)에 닿도록 center z=hz
            spheres, R = self._box_spheres(c, half)
            # #1 시작자세 팔과 최소 여유
            if self._min_gap_to_arm(arm, spheres, R, c, half) < self.obs_start_margin:
                continue
            # #2 goal과 최소 여유(최종 접근로 확보)
            if self._point_box_gap(goal, c, half) < self.obs_goal_margin:
                continue
            return c, half, spheres, R
        return None

    def _place_obstacles(self, start_tip, goal):
        """에피소드마다 장애물 k개(1~n_obstacles) 배치. rejection sampling으로
        #1 시작자세 팔여유·#2 goal여유를 보장하고, 앞 obs_path_relevant개는
        #3 start_tip→goal 경로 corridor 내(관련성)로 강제. 나머지는 완전 랜덤(다양성)."""
        m = self.model
        k = int(self.np_random.integers(1, self.n_obstacles + 1))
        n_path = min(self.obs_path_relevant, k)   # 경로 관련성 강제할 개수
        arm = self._arm_prims_world()             # 시작자세 팔(1회 캐시)
        self._obs_sph = []; self._active_obs = []
        for i, (ob, og) in enumerate(zip(self._obs_bodies, self._obs_geoms)):
            placed = self._sample_one_obstacle(arm, start_tip, goal, require_path=(i < n_path)) if i < k else None
            if placed is None:
                m.body_pos[ob] = [0, 0, 50.0]     # 비활성(멀리) — 미사용 or 배치 실패
                continue
            c, half, spheres, R = placed
            m.body_pos[ob] = c
            m.geom_size[og] = half
            self._obs_sph.extend((cc, R) for cc in spheres)
            self._active_obs.append(og)

    def _prim_pairs(self):
        """(닫힌점p1, 법선n, gap, body, link) 이터레이터.
        캡슐(boom/stick) = 해석적 capsule-sphere(박스 구근사) / 버킷박스 = mj_geomDistance(박스-박스)."""
        m, d = self.model, self.data
        for bid, la, lb, R, lnm in self._caps:
            Rm = d.xmat[bid].reshape(3, 3); p = d.xpos[bid]; A = p + Rm @ la; B = p + Rm @ lb
            for C, r_obs in self._obs_sph:
                p1 = self._segcl(A, B, C); dist = np.linalg.norm(C - p1)
                n = (C - p1) / (dist + 1e-9)
                yield p1, n, dist - R - r_obs, bid, lnm
        for g in self._bkt_geoms:
            cg = d.geom_xpos[g]
            hg = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]   # 버킷박스 월드 AABB half
            for og in self._active_obs:
                gap = mujoco.mj_geomDistance(m, d, g, og, 5.0, self._ft)
                # ccd(GJK/MPR)가 얇은 박스에서 수렴 실패 시 허위 0/음수 반환 → AABB-AABB 하한으로 보정.
                # 실제 표면거리 ≥ AABB간 거리(장애물은 축정렬). gap이 하한보다 작으면 허위값이므로 끌어올림.
                sep = np.abs(cg - d.geom_xpos[og]) - (hg + m.geom_size[og])
                lb = float(np.linalg.norm(np.maximum(sep, 0.0)) + min(float(np.max(sep)), 0.0))
                gap = max(gap, lb)
                p1 = self._ft[:3].copy(); p2 = self._ft[3:].copy()
                yield p1, (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9), gap, self._part04, "bucket"

    def _link_clearance(self):
        """링크별(boom/stick/bucket) 최근접 장애물 → (n̂ base상대, gap raw)."""
        res = {"boom": (np.zeros(3), 1e9), "stick": (np.zeros(3), 1e9), "bucket": (np.zeros(3), 1e9)}
        for p1, n, gap, bid, lnm in self._prim_pairs():
            if gap < res[lnm][1]:
                res[lnm] = (n.copy(), gap)
        return res

    def _ttc_severity(self):
        """min TTC(모든 프리미티브×장애물) → 지수커널 severity=exp(-TTC/τ). 관통=1, 접근없음=0."""
        min_ttc, min_gap = 1e9, 1e9
        for p1, n, gap, body, lnm in self._prim_pairs():
            min_gap = min(min_gap, gap)
            cl = float(self._pvel(p1, body) @ n)
            if cl > 1e-3 and gap > 0:
                min_ttc = min(min_ttc, gap / cl)
        self._last_min_gap = min_gap                     # 충돌판정용(≤0=관통) — step이 info로 노출
        if min_gap <= 0:
            return 1.0                                   # 관통 = 최대 위험
        if min_ttc >= 1e9:
            return 0.0                                   # 접근 없음
        return float(np.exp(-min_ttc / self.ttc_max))    # 지수커널: TTC=0→1, TTC 클수록 0

    def _reach_cloud(self, n, col_margin):
        """랜덤 관절 FK로 (r,z) 도달점 구름 수집 — 유효성 필터:
        (1) 지면 위(z>Z_MIN)·몸통 밖(r>R_MIN)  (2) 자기충돌 없음(버킷↔cabin/boom).
        결정론(시드0)이라 결과를 npz 캐시(XML 변경 시 자동 무효화)."""
        cache_dir = os.path.join(ROOT, "Model", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        xml_mtime = os.path.getmtime(XML)
        fp = os.path.join(cache_dir, f"reach_cloud_n{n}_m{col_margin}.npz")
        if os.path.exists(fp):
            z = np.load(fp)
            if float(z["xml_mtime"]) == xml_mtime:
                return z["rs"], z["zs"]
        rng = np.random.default_rng(0)                # envelope는 로봇 고유속성 → 고정 시드
        rs, zs = [], []
        for _ in range(n):
            self.data.qpos[:4] = rng.uniform(self.ctrl_lo, self.ctrl_hi)
            mujoco.mj_forward(self.model, self.data)
            t = self._tip(); r = float(np.hypot(t[0], t[1]))
            if t[2] > Z_MIN and r > R_MIN and not self._self_collision(col_margin):
                rs.append(r); zs.append(float(t[2]))
        rs, zs = np.array(rs), np.array(zs)
        np.savez(fp, rs=rs, zs=zs, xml_mtime=xml_mtime)
        return rs, zs

    def _build_reach_cells(self, n=50000, bins=50, min_count=1, col_margin=0.02):
        """init 1회: 랜덤 관절 FK로 도달 가능한 (r,z) envelope(2D 마스크) 구성.
        자기충돌·지면밑 config는 제거(_reach_cloud) → goal은 항상 FK-유효.
        + home 자세 bucket_tip(_tip_home)과 거기서의 도달거리 범위 → home-거리 커리큘럼 기준."""
        rs, zs = self._reach_cloud(n, col_margin)     # (충돌·무효 config 제거된) 도달점 구름
        r_hi, z_hi = 4.5, 4.5                         # trim 제거 
        self._r_edges = np.linspace(R_MIN, r_hi, bins + 1)
        self._z_edges = np.linspace(Z_MIN, z_hi, bins + 1)
        H, _, _ = np.histogram2d(rs, zs, bins=[self._r_edges, self._z_edges])
        self._mask = H >= min_count                   # 2D 도달 가능 마스크(빠른 reachability 조회)
        ii, jj = np.where(self._mask)
        rc = 0.5 * (self._r_edges[:-1] + self._r_edges[1:])
        zc = 0.5 * (self._z_edges[:-1] + self._z_edges[1:])
        self._cell_r, self._cell_z = rc[ii], zc[jj]
        self._reach_rmin, self._reach_rmax = float(self._cell_r.min()), float(self._cell_r.max())
        # home 자세 tip + home-거리 기준
        self.data.qpos[:4] = HOME; mujoco.mj_forward(self.model, self.data)
        self._tip_home = self._tip()
        rh = float(np.hypot(self._tip_home[0], self._tip_home[1])); zh = float(self._tip_home[2])
        th = float(np.arctan2(self._tip_home[1], self._tip_home[0]))
        # 셀별 home-거리: 최대(반대 방위)로 전체 범위, 최소로 폴백 안전 goal
        self._home_dmax = float(np.max(np.hypot(self._cell_r + rh, self._cell_z - zh)))
        k0 = int(np.argmin(np.hypot(self._cell_r - rh, self._cell_z - zh)))
        self._home_safe = np.array([self._cell_r[k0] * np.cos(th),
                                    self._cell_r[k0] * np.sin(th), self._cell_z[k0]])
        self._home_d0 = 0.6                            # 커리큘럼 시작 반경(작게=거의 안 돎)

    def _is_reachable(self, p):
        r = np.hypot(p[0], p[1])
        if r <= R_MIN or p[2] <= Z_MIN:
            return False
        i = int(np.searchsorted(self._r_edges, r)) - 1
        j = int(np.searchsorted(self._z_edges, p[2])) - 1
        if 0 <= i < self._mask.shape[0] and 0 <= j < self._mask.shape[1]:
            return bool(self._mask[i, j])
        return False

    def set_curriculum(self, frac):
        """거리 기준 커리큘럼. frac∈[0,1]: 0=시작 tip 근처(쉬움) → 1=전체.
        목표를 '시작 자세 tip에서의 거리'로 gating → 작은 reach부터 큰 선회까지 점진(stitching).
        frac=1이면 curr_r=None(전체 envelope). 학습 콜백이 성공률 따라 호출."""
        frac = float(np.clip(frac, 0.0, 1.0))
        self.curr_r = None if frac >= 1.0 else self._home_d0 + frac * (self._home_dmax - self._home_d0)

    def _sample_reachable_waypoint(self, center=None):
        """목표 waypoint 샘플 — 작업공간(Cartesian) 부피 균일.
        - curr_r=None: 전체 도달 envelope에서 3D 부피 균일(공간상 고르게 → 먼 goal도 비율만큼 등장 → 원거리 도달 학습 강화).
        - curr_r 설정: center(시작 tip) 주변 반경 curr_r 공 안 부피 균일(거리 커리큘럼).
        - goal_az_limit: 시작 전방(카메라 방향) ±limit로 방위각 제한(후면 목표 제외).
        셀 균일(원통 균일)은 근거리(작은 r)에 밀도가 몰려(∝1/r) 먼 goal이 과소표집 → 부피 균일로 교체."""
        c = self._tip_home if center is None else center
        lim, fwd = self.goal_az_limit, np.arctan2(c[1], c[0])   # 전방 = 시작 tip 방위
        r_lo, r_hi = self._reach_rmin, self._reach_rmax
        z_lo, z_hi = float(self._z_edges[0]), float(self._z_edges[-1])

        def _workspace_uniform():   # 도달 envelope 내 부피 균일: 원기둥 부피 균일 샘플 → reachability 리젝션
            for _ in range(128):
                r = float(np.sqrt(self.np_random.uniform(r_lo ** 2, r_hi ** 2)))  # 면적가중 → 수평 균일
                z = float(self.np_random.uniform(z_lo, z_hi))                     # z 균일
                th = (fwd + self.np_random.uniform(-lim, lim)) if lim is not None \
                    else self.np_random.uniform(-np.pi, np.pi)                    # 방위 균일(전방 ±lim)
                p = np.array([r * np.cos(th), r * np.sin(th), z])
                if self._is_reachable(p):
                    return p
            return self._home_safe.copy()                    # 폴백(항상 reachable·전방)

        if self.curr_r is None:                              # 전체 범위(작업공간 부피 균일)
            return _workspace_uniform()
        for _ in range(64):                                  # 거리 커리큘럼(공 부피 균일: U^(1/3))
            u = self.np_random.normal(size=3); u /= (np.linalg.norm(u) + 1e-9)
            rad = self.curr_r * self.np_random.uniform() ** (1.0 / 3.0)
            cand = c + rad * u
            if lim is not None:                              # 전방 ±limit 밖이면 reject
                dth = np.arctan2(cand[1], cand[0]) - fwd
                if abs(np.arctan2(np.sin(dth), np.cos(dth))) > lim:
                    continue
            if self._is_reachable(cand):
                return cand
        return _workspace_uniform()                          # 폴백도 작업공간 균일(원통 균일 잔재 제거)

    def _get_obs(self):
        d = self.data
        q = (d.qpos[:4] - self.ctrl_mid) / self.ctrl_half          # qpos → [-1,1]
        qd = np.clip(d.qvel[:4] / self.v_max, -1.0, 1.0)           # qvel → 관절별 v_max 정규화+clip
        obs = np.concatenate([q, qd]).astype(np.float32)
        if self.obstacle_training:                                 # 링크별 clearance(n̂3+gap1)×3
            cl = self._link_clearance(); extra = []
            for lnm in ("boom", "stick", "bucket"):
                n, gap = cl[lnm]
                gap_norm = float(np.clip(gap / self.pos_scale, -1.0, 2.0))   # goal과 동일 스케일 정규화
                if gap_norm >= 2.0:                                # 너무 멀면 "장애물 없음"
                    n = np.zeros(3)
                extra.extend([*n, gap_norm])
            obs = np.concatenate([obs, np.array(extra, np.float32)]).astype(np.float32)
        return dict(observation=obs,
                    achieved_goal=(self._tip() / self.pos_scale).astype(np.float32),    # tip 정규화
                    desired_goal=(self._desired / self.pos_scale).astype(np.float32))    # goal 정규화

    # ── gym API ──
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # 시작 자세: random_start면 관절 랜덤(stitching·일반화), 아니면 home 고정
        if self.random_start:
            self.data.qpos[:4] = self.np_random.uniform(self.ctrl_lo, self.ctrl_hi)
        else:
            self.data.qpos[:4] = HOME
        mujoco.mj_forward(self.model, self.data)
        start_tip = self._tip()
        # 목표 waypoint 먼저 결정(장애물이 goal·경로를 참조하므로): options로 지정 or 시작 tip 기준 샘플
        goal = None
        if options is not None and options.get("goal") is not None:
            goal = np.asarray(options["goal"], dtype=float)
        self._desired = goal if goal is not None else self._sample_reachable_waypoint(center=start_tip)
        if self.obstacle_training:                          # goal·시작팔 회피 + 경로 관련성 배치 후 반영
            self._place_obstacles(start_tip, self._desired)
            mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qpos[:4]
        self._steps = 0
        self._prev_action = np.zeros(4, dtype=np.float32)   # action-rate penalty 기준(첫 스텝은 0 대비)
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        tgt = np.clip(self.data.qpos[:4] + a * self.dmax, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[:] = tgt
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))   # 정규화 거리
        sw, rc, _su = R.decomposed_terms(obs["achieved_goal"], obs["desired_goal"], self.eps_n)
        info = {"is_success": d < self.eps_n, "dist": d * self.pos_scale,        # dist는 원본 m
                "swing": float(sw), "reach": float(rc)}                          # 보상 항 모니터링용
        # action-rate(부드러운 제어) penalty: 직전 action과의 변화량 mean((a-a_prev)²)/4 ∈[0,1].
        #   (a-a_prev)∈[-2,2] → 제곱 [0,4] → /4 정규화. goal 독립 → info 경유로 HER relabel에도 유효.
        info["rate_pen"] = float(np.mean((a - self._prev_action) ** 2) / 4.0)
        self._prev_action = a                                                    # 다음 스텝 변화량 기준
        if self.obstacle_training:                                               # 장애물 TTC severity(goal 독립 → HER 안전)
            info["ttc_severity"] = self._ttc_severity()
            info["min_gap"] = float(self._last_min_gap)                           # ≤0=관통(collision 판정용, 평가지표)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        terminated = d < self.eps_n
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_fn(achieved_goal, desired_goal, info, self.eps_n)   # 정규화 eps

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
