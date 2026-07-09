"""
SAC + HER 학습 엔트리 (waypoint reaching). config.yaml 기반.
실행: conda run -n seok python Model/train.py [config.yaml] [--run name] [--steps N]
산출: Model/checkpoints/<run>.zip, Model/logs/<run>/(TB+metrics.csv), Evaluate/plots/<run>_curve.png
"""
import os
import sys
import argparse
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
from excavator_env import ExcavatorRHEREnv
from callbacks import (TrainMetricsCallback, SaveLatestCallback, CurriculumCallback,
                       FullTaskEvalCallback, VideoRecorderCallback, WttcRampCallback, save_curves)


def make_env(cfg, seed=None):
    e = ExcavatorRHEREnv(eps=cfg["eps"], max_steps=cfg["max_steps"], frame_skip=cfg["frame_skip"],
                         reward=cfg.get("reward", "sparse"),
                         random_start=cfg.get("random_start", False),
                         normalize_goal=cfg.get("normalize_goal", True),
                         goal_azimuth_limit_deg=cfg.get("goal_azimuth_limit_deg", None),
                         obstacle_training=cfg.get("obstacle_training", False),
                         n_obstacles=cfg.get("n_obstacles", 3),
                         obs_len_range=cfg.get("obs_len_range", (0.3, 0.9)),
                         obs_wid_range=cfg.get("obs_wid_range", (0.3, 0.9)),
                         obs_hgt_range=cfg.get("obs_hgt_range", (0.3, 0.9)),
                         obs_start_margin=cfg.get("obs_start_margin", 0.30),
                         obs_goal_margin=cfg.get("obs_goal_margin", 0.40),
                         obs_path_relevant=cfg.get("obs_path_relevant", 1),
                         obs_path_corridor=cfg.get("obs_path_corridor", 0.6),
                         obs_place_tries=cfg.get("obs_place_tries", 20),
                         ttc_max=cfg.get("ttc_max", 2.0),
                         w_ttc=cfg.get("w_ttc", 0.3))
    if seed is not None:
        e.reset(seed=seed)
    return e


def train(cfg, resume=None):
    run = cfg["run_name"]
    logdir = os.path.join(ROOT, "Model", "logs", run); os.makedirs(logdir, exist_ok=True)
    ckpt = os.path.join(ROOT, "Model", "checkpoints", f"{run}.zip")
    plot = os.path.join(ROOT, "Evaluate", "plots", f"{run}_curve.png")
    csvp = os.path.join(logdir, "metrics.csv")

    raw_env = make_env(cfg)
    env = Monitor(raw_env)
    eval_env = make_env(cfg)
    if resume:                              # 이어학습: 모델(정책·Q·옵티마이저) 로드
        model = SAC.load(resume, env=env, tensorboard_log=logdir)
        rb = os.path.splitext(resume)[0] + "_replay.pkl"
        if os.path.exists(rb):              # replay buffer가 같이 저장돼 있으면 복원
            model.load_replay_buffer(rb); print(f"[{run}] replay buffer 복원: {rb}")
        else:
            print(f"[{run}] ⚠️ replay buffer 없음 → 빈 버퍼로 재시작(learning_starts 재충전)")
        print(f"[{run}] resume from {resume} (num_timesteps={model.num_timesteps})")
    else:
        model = SAC("MultiInputPolicy", env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(n_sampled_goal=cfg["n_sampled_goal"],
                                              goal_selection_strategy=cfg["goal_selection_strategy"]),
                    buffer_size=cfg["buffer_size"], learning_starts=cfg["learning_starts"],
                    batch_size=cfg["batch_size"], gamma=cfg["gamma"],
                    learning_rate=cfg["learning_rate"], seed=cfg["seed"],
                    target_entropy=cfg.get("target_entropy", "auto"),
                    tensorboard_log=logdir, verbose=0)
    cb = TrainMetricsCallback(eval_env, eval_freq=cfg["eval_freq"], n_eval=cfg["n_eval"],
                              csv_path=csvp, verbose=1)
    saver = SaveLatestCallback(ckpt, save_freq=cfg.get("save_freq", 5000), verbose=1)
    # ★ 고정 full-task 평가: 커리큘럼과 무관한 별도 env(curr_r=None 유지) + 고정 seed + 큰 n_eval
    full_eval_env = make_env(cfg)         # CurriculumCallback에 넣지 않음 → 전체 task 유지
    full_cb = FullTaskEvalCallback(full_eval_env, eval_freq=cfg["eval_freq"],
                                   n_eval=cfg.get("n_eval_full", 50),
                                   csv_path=os.path.join(logdir, "fulltask.csv"),
                                   fail_csv_path=os.path.join(logdir, "failures.csv"),
                                   eval_entropy=cfg.get("eval_entropy", True), verbose=1)
    cbs = [cb, full_cb, saver]
    if cfg.get("curriculum", False):     # r 기준 적응 커리큘럼(원통 샘플)
        cbs.insert(1, CurriculumCallback([raw_env, eval_env], cb,
                                         start=cfg.get("curr_start", 0.3),
                                         inc=cfg.get("curr_inc", 0.2),
                                         up=cfg.get("curr_up", 0.7),
                                         te_boost=cfg.get("curr_te_boost", 0.0),
                                         te_halflife=cfg.get("curr_te_halflife", 5000), verbose=1))
    if cfg.get("w_ttc_ramp"):            # w_ttc를 [start, end, n_steps]로 선형 램프(reaching 먼저)
        s, e, n = cfg["w_ttc_ramp"]
        cbs.append(WttcRampCallback(s, e, n, verbose=1))
        print(f"[{run}] w_ttc ramp {s}→{e} over {n} steps")
    if cfg.get("video_freq", 0) > 0:     # N스텝마다 롤아웃 영상 → TensorBoard
        cbs.append(VideoRecorderCallback(eval_env, render_freq=cfg["video_freq"], verbose=1))
    print(f"[{run}] task=waypoint curriculum={cfg.get('curriculum', False)} "
          f"steps={cfg['total_steps']} device={model.device}")
    print(f"[{run}] 라이브 관찰: 다른 터미널에서 "
          f"conda run -n seok python Evaluate/visualize.py --ckpt {ckpt}")
    model.learn(total_timesteps=cfg["total_steps"], callback=cbs, progress_bar=True,
                reset_num_timesteps=not resume)         # resume이면 스텝 카운터 이어감
    model.save(ckpt)
    model.save_replay_buffer(os.path.splitext(ckpt)[0] + "_replay.pkl")   # 다음 resume용 버퍼 저장
    save_curves(cb.records, plot, title=f"{run} (waypoint)")
    final = cb.records[-1] if cb.records else (0, 0, 0, 0)
    print(f"[{run}] 완료 → ckpt={ckpt} | 최종 success={final[1]:.2f} dist={final[2]:.3f}")
    return cb.records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--run"); ap.add_argument("--steps", type=int)
    ap.add_argument("--resume")   # 이어학습할 체크포인트 경로(예 Model/checkpoints/target_entropy.zip)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.run: cfg["run_name"] = a.run
    if a.steps: cfg["total_steps"] = a.steps
    train(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
