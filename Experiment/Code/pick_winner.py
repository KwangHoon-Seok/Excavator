"""1M 학습 4버전 중 최종 safe_success 최고 승자의 run 이름(gcrl_vN_...) 출력."""
import csv
import os

LOGDIR = "/home/elicer/Excavator/Model/logs/GCRL_compare"
RUNS = ["gcrl_v1_startX_dist", "gcrl_v2_startO_dist", "gcrl_v3_startX_auto", "gcrl_v4_startO_auto"]

best = None
for r in RUNS:
    p = os.path.join(LOGDIR, r, "fulltask.csv")
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p)))
    if not rows:
        continue
    ss = float(rows[-1].get("safe_success", 0))
    if best is None or ss > best[1]:
        best = (r, ss)
print(best[0] if best else "gcrl_v3_startX_auto")
