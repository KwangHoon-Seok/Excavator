#!/bin/bash
# curriculum·target-entropy boosting 끈 baseline GCRL 3 seed 병렬 학습(각 2M).
# progress_bar off(rich 충돌 회피) + 스태거 시작(OSMesa/learn 초기화 경합 회피).
cd /home/elicer/Excavator
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PROGRESS_BAR=0
SC=/tmp/claude-10000/-home-elicer-Excavator/eb5fd0dd-a249-43df-b7d6-37a872d577d3/scratchpad
echo "[launch] $(date) baseline 3seed 시작"
for sd in 1 2 3; do
  rm -rf Model/logs/GCRL_baseline/nocurric_s${sd} Model/checkpoints/GCRL_baseline/nocurric_s${sd}* 2>/dev/null
  python3 -u Model/Version4/train.py Model/Version4/config_baseline_s${sd}.yaml > $SC/base_s${sd}.log 2>&1 &
  echo "  seed$sd -> PID $!  ($(date +%H:%M:%S))"
  sleep 20     # 스태거: 다음 seed 시작 전 이 seed가 learn() 진입하도록
done
wait
echo "[done] $(date) baseline 3seed 완료"
