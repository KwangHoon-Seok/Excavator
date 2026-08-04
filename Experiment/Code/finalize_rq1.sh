#!/bin/bash
cd /home/elicer/Excavator/Experiment/Code
export MUJOCO_GL=osmesa
R1=out/cells/Qwen3.5-27B__rgb_only__id__rq1_rgb_only/results.csv
R2=out/cells/Qwen3.5-27B__height_only__id__rq1_height_only/results.csv
echo "[finalize] waiting for both cells..."
while [ ! -f "$R1" ] || [ ! -f "$R2" ]; do sleep 60; done
echo "[finalize] both cells done. aggregating..."
python3 aggregate_rqs.py
echo "[finalize] generating videos (GPU0)..."
CUDA_VISIBLE_DEVICES=0 python3 make_rq1_videos.py
echo "[finalize] DONE" && date
