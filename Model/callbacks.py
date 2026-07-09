"""
학습 추이 기록 콜백 + 그래프 저장.
- 주기적으로 결정론 eval N에피소드 → 성공률·평균거리 기록(step별).
- metrics.csv 저장 + matplotlib로 reward/success vs step PNG 저장.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import torch as th
import reward as R
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Video


class WttcRampCallback(BaseCallback):
    """장애물 회피 보상 가중 w_ttc를 start→end로 선형 램프(0~n_steps).
    reward_decomposed가 참조하는 모듈 전역 R.W_TTC를 갱신(모든 env 공유). 초반 낮은 값으로
    obstacle obs를 살리며 reaching 먼저 → 점차 회피 강화. safe/collision은 기하량이라 영향 없음."""
    def __init__(self, start, end, n_steps, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.n = float(start), float(end), max(1, int(n_steps))
        self._last = None

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.n)
        R.W_TTC = self.start + (self.end - self.start) * frac
        if self.verbose and (self._last is None or self.num_timesteps - self._last >= 50000):
            print(f"[w_ttc ramp] @ {self.num_timesteps}: W_TTC={R.W_TTC:.3f}")
            self._last = self.num_timesteps
        return True


def _goal_sphere(scn, pos):
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.25, 0, 0.]), np.asarray(pos, float), np.eye(3).flatten(),
                        np.array([0.1, 0.9, 0.1, 0.8], np.float32))
    scn.ngeom += 1


class VideoRecorderCallback(BaseCallback):
    """render_freq 스텝마다 결정론 롤아웃 1에피소드를 offscreen 렌더 → TensorBoard에 영상 기록.
    TB의 SCALARS가 아니라 IMAGES 탭에 'rollout/video'로 뜸. (moviepy 필요)"""
    def __init__(self, eval_env, render_freq, fps=20, size=(360, 480), max_frames=200, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.fps = fps
        self.h, self.w = size
        self.max_frames = max_frames
        self._r = None

    def _setup(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.eval_env.model, self.h, self.w)
            self._cam = mujoco.MjvCamera(); self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0, 1.0, 0.3]; self._cam.distance = 7.0
            self._cam.azimuth = 135; self._cam.elevation = -20
        return self._r

    def _rollout_frames(self):
        r = self._setup()
        frames = []
        # 스텝 기반 롤링 시드 → 매 영상이 서로 다른 씬(현재 커리큘럼 레벨). 공유 eval_env가
        # 직전 고정시드 평가로 오염돼 매번 같은 씬이 나오던 문제 방지(900000 네임스페이스로 eval 시드와 분리).
        obs, _ = self.eval_env.reset(seed=900_000 + int(self.num_timesteps))
        done = False
        while not done and len(frames) < self.max_frames:
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.eval_env.step(a)
            r.update_scene(self.eval_env.data, self._cam)
            _goal_sphere(r.scene, self.eval_env._desired)
            frames.append(r.render())
            done = term or trunc
        return np.asarray(frames, dtype=np.uint8)            # (T,H,W,3)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_freq == 0:
            f = self._rollout_frames()
            vid = th.from_numpy(f.transpose(0, 3, 1, 2)[None])  # (1,T,C,H,W)
            self.logger.record("rollout/video", Video(vid, fps=self.fps),
                               exclude=("stdout", "log", "json", "csv"))
            if self.verbose:
                print(f"[video] TB 영상 기록 @ {self.num_timesteps} ({len(f)} frames)")
        return True


class TrainMetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval=10, csv_path=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.csv_path = csv_path
        self.records = []  # (step, success_rate, mean_dist, ep_rew_mean)

    def _ep_rew_mean(self):
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            return float(np.mean([e["r"] for e in buf]))
        return np.nan

    def _evaluate(self):
        succ, dists = 0, []
        sw_sum, re_sum, rt_sum, n = 0.0, 0.0, 0.0, 0   # 보상 항 누적(스텝 평균용)
        for i in range(self.n_eval):
            obs, _ = self.eval_env.reset(seed=10_000 + i)
            done = False
            info = {}
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(a)
                sw_sum += info.get("swing", 0.0); re_sum += info.get("reach", 0.0)
                rt_sum += info.get("rate_pen", 0.0); n += 1
                done = term or trunc
            succ += int(info.get("is_success", False))
            dists.append(info.get("dist", np.nan))
        m = max(n, 1)
        self._terms = (sw_sum / m, re_sum / m, rt_sum / m)  # (swing, reach, rate) 평균
        return succ / self.n_eval, float(np.nanmean(dists))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            sr, md = self._evaluate()
            rew = self._ep_rew_mean()
            self.records.append((self.num_timesteps, sr, md, rew))
            self.logger.record("eval/success_rate", sr)   # TB에 기록
            self.logger.record("eval/mean_dist", md)
            self.logger.record("reward/succ", sr)                 # 보상 항 모니터링(succ=성공률)
            self.logger.record("reward/swing", self._terms[0])    # swing 항 평균(0=정렬, 낮을수록↑)
            self.logger.record("reward/reach", self._terms[1])    # reach 항 평균(0=도달, 낮을수록↑)
            self.logger.record("reward/rate_pen", self._terms[2])  # action 변화량 평균(낮을수록 부드러움)
            if self.verbose:
                print(f"[step {self.num_timesteps}] success={sr:.2f} dist={md:.3f} ep_rew={rew:.2f}")
            if self.csv_path:
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["step", "success_rate", "mean_dist", "ep_rew_mean"])
                    w.writerows(self.records)
        return True


class FullTaskEvalCallback(BaseCallback):
    """★ 고정 full-task 평가 — 커리큘럼과 무관하게 '최종 target task(전체 범위)'에서 성공률 측정.
    env는 curr_r=None(전체)로 고정하고 CurriculumCallback이 건드리지 않는 별도 인스턴스여야 함.
    고정 seed + 큰 n_eval → config 간 비교 가능한 신뢰 지표(eval_fulltask/*).

    디버깅 계측(모두 이 eval에서 수집):
      1) eval_region/succ_* : goal을 방위(front/side/wide)·거리(near/mid/far)·높이(lowz/highz)로
         분류한 구역별 성공률 → '뒤/먼/높은 goal 실패'를 국소화.
      2) policy/entropy_actual vs policy/target_entropy : 실제 −E[logπ] vs setpoint(α 붕괴 감지).
      3) act/<joint>_mag·_sat : 관절별 |action| 평균·포화율(|a|>0.95) → swing 포화·제어 사멸 감지.
      4) fail_csv : 실패 goal의 (r,z,방위,시작거리,최종dist) 덤프 → 실패 구조 산점도.
      5) eval_fulltask/steps_to_success : 성공 에피소드 평균 스텝(도달 효율)."""
    def __init__(self, env, eval_freq=10000, n_eval=50, seed0=50000, csv_path=None,
                 fail_csv_path=None, eval_entropy=True, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.seed0 = seed0
        self.csv_path = csv_path
        self.fail_csv_path = fail_csv_path
        self.eval_entropy = eval_entropy     # 정책 엔트로피 추정(스텝당 forward 추가) on/off
        self.records = []
        self._failures = []          # (step, r, z, az_deg, dist_from_start, final_dist)

    def _policy_entropy(self, obs):
        """현재 정책의 상태별 엔트로피 추정 −E[logπ(a|s)] (target_entropy와 직접 비교용)."""
        try:
            obs_t = self.model.policy.obs_to_tensor(obs)[0]
            with th.no_grad():
                _, log_prob = self.model.actor.action_log_prob(obs_t)
            return float(-log_prob.mean().item())
        except Exception:
            return np.nan

    @staticmethod
    def _bins(az_deg, d0, z):
        azb = "front" if az_deg < 45 else ("side" if az_deg < 90 else "wide")
        db = "near" if d0 < 2.0 else ("mid" if d0 < 3.5 else "far")
        zb = "lowz" if z < 1.0 else "highz"
        return azb, db, zb

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self.env.set_curriculum(1.0)   # 항상 전체 task(curr_r=None) 보장
        succ, dists = 0, []
        n_coll, n_safe = 0, 0          # 장애물 회피 지표: 충돌(관통) 에피소드 수, 안전성공(도달+무충돌) 수
        region = {}                    # bin 이름 -> [n, n_succ]
        ent_sum, ent_n = 0.0, 0
        amag, sat, astep = np.zeros(4), np.zeros(4), 0
        steps_succ = []
        JOINTS = ["swing", "boom", "stick", "bucket"]
        for i in range(self.n_eval):
            obs, _ = self.env.reset(seed=self.seed0 + i)   # 고정 seed → 재현·비교 가능
            start = self.env._tip()                        # 시작 tip(방위/거리 기준)
            fwd = np.arctan2(start[1], start[0])
            g = np.asarray(self.env._desired, float)
            dth = np.arctan2(g[1], g[0]) - fwd
            az_deg = abs(np.degrees(np.arctan2(np.sin(dth), np.cos(dth))))   # 전방 대비 |방위|
            d0 = float(np.linalg.norm(g - start))          # 시작 tip에서 goal 거리
            gr, gz = float(np.hypot(g[0], g[1])), float(g[2])
            done = False; info = {}; steps = 0; ep_collided = False
            while not done:
                if self.eval_entropy:
                    e = self._policy_entropy(obs)
                    if np.isfinite(e):
                        ent_sum += e; ent_n += 1
                a, _ = self.model.predict(obs, deterministic=True)
                amag += np.abs(a); sat += (np.abs(a) > 0.95); astep += 1
                obs, r, term, trunc, info = self.env.step(a); steps += 1
                if info.get("min_gap", 1e9) <= 0.0:        # 링크가 장애물 관통 → 이 에피소드 충돌
                    ep_collided = True
                done = term or trunc
            ok = int(info.get("is_success", False))
            succ += ok; fd = float(info.get("dist", np.nan)); dists.append(fd)
            if ep_collided:
                n_coll += 1
            if ok and not ep_collided:                     # 안전성공 = 도달 AND 무충돌
                n_safe += 1
            if ok:
                steps_succ.append(steps)
            else:
                self._failures.append((self.num_timesteps, gr, gz, az_deg, d0, fd))
            for b in self._bins(az_deg, d0, gz):
                s = region.setdefault(b, [0, 0]); s[0] += 1; s[1] += ok

        sr, md = succ / self.n_eval, float(np.nanmean(dists))
        safe_r, coll_r = n_safe / self.n_eval, n_coll / self.n_eval
        self.records.append((self.num_timesteps, sr, md, safe_r, coll_r))
        self.logger.record("eval_fulltask/success_rate", sr)
        self.logger.record("eval_fulltask/mean_dist", md)
        self.logger.record("eval_fulltask/collision_rate", n_coll / self.n_eval)      # 관통 에피소드 비율
        self.logger.record("eval_fulltask/safe_success_rate", n_safe / self.n_eval)   # 도달+무충돌(헤드라인)
        for b, (nb, sb) in region.items():                        # 1) 구역별 성공률
            self.logger.record(f"eval_region/succ_{b}", sb / nb)
        if steps_succ:                                            # 5) 도달 효율
            self.logger.record("eval_fulltask/steps_to_success", float(np.mean(steps_succ)))
        if ent_n:                                                 # 2) 정책 엔트로피 vs setpoint
            self.logger.record("policy/entropy_actual", ent_sum / ent_n)
        self.logger.record("policy/target_entropy", float(self.model.target_entropy))
        if astep:                                                # 3) 관절별 action 포화/크기
            for j, nm in enumerate(JOINTS):
                self.logger.record(f"act/{nm}_mag", float(amag[j] / astep))
                self.logger.record(f"act/{nm}_sat", float(sat[j] / astep))
        if self.verbose:
            rg = " ".join(f"{b}={s[1]}/{s[0]}" for b, s in sorted(region.items()))
            print(f"[fulltask @ {self.num_timesteps}] success={sr:.2f} safe={n_safe/self.n_eval:.2f} "
                  f"coll={n_coll/self.n_eval:.2f} dist={md:.3f} "
                  f"ent={ent_sum/max(ent_n,1):.2f}/te={float(self.model.target_entropy):.2f} | {rg}")
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "fulltask_success", "fulltask_mean_dist", "safe_success", "collision_rate"])
                w.writerows(self.records)
        if self.fail_csv_path and self._failures:                # 4) 실패 goal 덤프
            with open(self.fail_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "r", "z", "az_deg", "dist_from_start", "final_dist"])
                w.writerows(self._failures)
        return True


class CurriculumCallback(BaseCallback):
    """r 기준 적응 커리큘럼: 현재 난이도에서 성공률이 up 이상이면 허용 반경을 inc만큼 넓힘.
    envs(train·eval) 모두 set_curriculum 적용 → 쉬운 근거리부터 점차 먼 waypoint까지."""
    def __init__(self, envs, metrics_cb, start=0.3, inc=0.2, up=0.7,
                 te_boost=0.0, te_halflife=20000, verbose=0):
        super().__init__(verbose)
        self.envs = envs                 # set_curriculum를 가진 env 리스트(train raw, eval)
        self.metrics_cb = metrics_cb     # 최신 성공률 참조용
        self.frac, self.inc, self.up = start, inc, up
        # 적응적 탐색: target_entropy = base + te_boost · min(time_decay, success_g)
        #   time_decay = 0.5^(상승후경과 / halflife)  → 시간 상한(무조건 감소, explore-forever 방지)
        #   success_g  = (up−success)/up             → 성공률↑이면 빨리 exploit
        #   min → 둘 중 먼저 "탐색 줄여"라는 쪽을 따름(안전+easy 적응)
        self.te_boost = float(te_boost)
        self.te_halflife = float(te_halflife)
        self._base_te = None
        self._te = None
        self._last_advance = 0           # 마지막 커리큘럼 상승 시점(스텝)
        self._last_checked = -1

    def _apply(self):
        for e in self.envs:
            e.set_curriculum(self.frac)

    def _on_training_start(self) -> None:
        self._apply()
        if self.te_boost:                # 기존 target_entropy를 복구 목표(base)로 기록
            self._base_te = float(self.model.target_entropy)
            self._te = self._base_te
        if self.verbose:
            print(f"[curriculum] 시작 frac={self.frac:.2f} (base target_entropy={self._base_te})")

    def _on_step(self) -> bool:
        if self.te_boost and self._te is not None:      # 매 스텝 기록 → target_entropy 스케줄 dense 시각화
            self.logger.record("curriculum/target_entropy", float(self._te))
        rec = self.metrics_cb.records
        if rec and rec[-1][0] != self._last_checked:   # metrics_cb가 새 eval 기록했을 때만
            self._last_checked = rec[-1][0]
            success = rec[-1][1]
            advanced = self.frac < 1.0 and success >= self.up
            if advanced:
                self.frac = min(1.0, self.frac + self.inc)
                self._apply()
                self._last_advance = self.num_timesteps
                if self.verbose:
                    print(f"[curriculum] success={success:.2f} → frac↑ {self.frac:.2f}")
            # target_entropy 적응 갱신: 상승 직후엔 full, 그 외엔 min(time_decay, success_g)
            if self.te_boost:
                if advanced:
                    bf = 1.0
                else:
                    t_adv = self.num_timesteps - self._last_advance
                    time_decay = 0.5 ** (t_adv / self.te_halflife)
                    success_g = float(np.clip((self.up - success) / self.up, 0.0, 1.0))
                    bf = min(time_decay, success_g)
                self._te = self._base_te + self.te_boost * bf
                self.model.target_entropy = self._te
            cr = self.envs[0].curr_r                                 # frac=1이면 None(전체)
            self.logger.record("curriculum/frac", self.frac)        # TB에 기록
            self.logger.record("curriculum/curr_r", float(cr) if cr is not None else float(self.envs[0]._home_dmax))
        return True


class SaveLatestCallback(BaseCallback):
    """주기적으로 모델을 '같은 경로'에 덮어써 저장 → 다른 터미널 visualize.py가 최신 정책 로드 가능."""
    def __init__(self, save_path, save_freq=5000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"[save] {self.save_path} @ {self.num_timesteps}")
        return True


def save_curves(records, out_png, title=""):
    if not records:
        return
    rec = np.array(records, dtype=float)
    step, sr, md, rew = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(step, sr, "-o"); ax[0].set_title("success rate"); ax[0].set_ylim(-0.05, 1.05)
    ax[1].plot(step, md, "-o", color="tab:orange"); ax[1].set_title("mean final dist (m)")
    ax[2].plot(step, rew, "-o", color="tab:green"); ax[2].set_title("ep reward mean")
    for a in ax:
        a.set_xlabel("timestep"); a.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_png, dpi=110); plt.close(fig)
