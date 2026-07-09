"""
Reward 함수 — 모두 (achieved_goal, desired_goal, info, eps)만으로 계산.

HER 제약: HER는 transition의 desired_goal을 '실제 달성한 상태'로 relabel한 뒤
  같은 함수로 보상을 재계산한다. 따라서 보상은 '절대 상태/시간'이 아니라
  오직 (달성 goal, 목표 goal)의 관계로만 정의돼야 한다.
  (action·energy 같은 transition 속성은 goal과 무관 → info에 담아 쓰면 relabel에도 유효.)

입출력: 스칼라(env.step) / 배치(HER, ag·dg가 (B,3)) 모두 처리(axis=-1, broadcasting).
"""
import numpy as np

dense_w = 0.3

def _dist(ag, dg):
    a, d = np.asarray(ag, np.float32), np.asarray(dg, np.float32)
    return np.linalg.norm(a[..., :3] - d[..., :3], axis=-1)


def _field(info, key, default=0.0):
    """info(스칼라 dict[env] 또는 batch array-of-dict[HER])에서 스칼라 항 추출.
    transition 속성(action·qvel penalty)은 goal 독립 → info 경유로 HER relabel에도 유효.
    env가 아직 key를 안 넣었으면 default(0) → 해당 penalty 무효(안전 폴백)."""
    if isinstance(info, dict):
        return np.float32(info.get(key, default))
    return np.array([(i.get(key, default) if isinstance(i, dict) else default) for i in info],
                    dtype=np.float32)


# ── Sparse (HER 정석) ──────────────────────────────────────────────
def sparse(ag, dg, info, eps):
    """목표 도달(‖ag-dg‖ ≤ eps)이면 0, 아니면 -1.

    설계 의도:
    - 도달 전엔 매 스텝 -1 → 빨리 도달할수록 누적보상↑ (최단 도달 유도).
    - shaping이 전혀 없어 reward hacking 불가(목표 도달만이 유일한 개선).
    - 그냥 쓰면 보상이 거의 항상 -1이라 학습 신호가 희박하지만,
      HER가 실패 경로의 goal을 '달성 지점'으로 바꾸면 그 transition이 0(성공)으로
      되살아나 신호가 생긴다 → **sparse + HER**가 정확히 이 문제를 푼다.
    """
    return -(_dist(ag, dg) > eps).astype(np.float32)


# ── Dense (shaping: 거리 패널티) ───────────────────────────────────
def dense(ag, dg, info, eps):
    d = _dist(ag, dg)
    dense_r = tanh_kernel(d)
    return dense_r

def sparse_dense(ag, dg, info, eps):
    sparse_r = sparse(ag, dg, info, eps)
    dense_r = dense(ag, dg, info, eps)
    return sparse_r + dense_w * dense_r


# tanh kernel
def tanh_kernel(d):
    return np.tanh(-d)


# ── Decomposed (sparse + swing(방위) + reach(평면)) ─────────────────
REACH_NORM = 1.0   # reach항 정규화 상수(폴백). env init이 envelope 측정값으로 덮어씀(R.REACH_NORM)
W_TTC = 0.5       # 장애물 TTC severity 페널티 가중(obstacle_training). env init이 cfg값으로 덮어씀


def _cyl_err(ag, dg):
    """정규화 ag,dg(원통 재표현) → (θ_err[0,π], r_err, z_err). 전부 ag,dg만 사용(HER 안전)."""
    a, d = np.asarray(ag, np.float32), np.asarray(dg, np.float32)
    dth = np.arctan2(a[..., 1], a[..., 0]) - np.arctan2(d[..., 1], d[..., 0])
    theta_err = np.abs(np.arctan2(np.sin(dth), np.cos(dth)))          # wrap → [0,π]
    r_err = np.abs(np.hypot(a[..., 0], a[..., 1]) - np.hypot(d[..., 0], d[..., 1]))
    z_err = np.abs(a[..., 2] - d[..., 2])
    return theta_err, r_err, z_err


def reward_decomposed(ag, dg, info, eps, w_s=0.5, w_r=0.5, w_rate=0.1):
    sparse_r = sparse(ag, dg, info, eps)
    theta_err, r_err, z_err = _cyl_err(ag, dg)
    swing_r = (theta_err / np.pi) ** 2
    reach_r = np.clip((r_err ** 2 + z_err ** 2) / REACH_NORM, 0.0, 1.0)
    # action-rate(부드러운 제어) penalty — env.step이 info["rate_pen"]=mean((a-a_prev)²)/4∈[0,1] 제공.
    #   크기(a²)가 아닌 변화량을 벌 → 빠른 이동은 허용, 스텝간 덜컹거림(채터링)만 억제
    #   → far-goal 도달과 충돌 안 함. goal 독립 → info 경유로 HER relabel에도 유효.
    rate_pen = _field(info, "rate_pen")
    # 장애물 TTC severity penalty — env.step이 info["ttc_severity"]∈[0,1] 제공(obstacle_training).
    #   충돌 임박할수록 1 → 페널티↑. goal 독립 → HER relabel에도 유효. 없으면 0(폴백).
    ttc_pen = _field(info, "ttc_severity")
    return sparse_r - w_s * swing_r - w_r * reach_r - w_rate * rate_pen - W_TTC * ttc_pen



def decomposed_terms(ag, dg, eps):
    """모니터링용 — reward_decomposed의 (swing, reach, succ) 항을 [0,1]로 반환."""
    theta_err, r_err, z_err = _cyl_err(ag, dg)
    swing = (theta_err / np.pi) ** 2
    reach = np.clip((r_err ** 2 + z_err ** 2) / REACH_NORM, 0.0, 1.0)
    succ = (_dist(ag, dg) <= eps).astype(np.float32)
    return swing, reach, succ


