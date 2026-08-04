#!/bin/bash
# 전체 RQ 실행 오케스트레이션 (brief 출력 + 2×A100 병렬). 재개 가능(완료 셀 skip).
cd /home/elicer/Excavator
export MUJOCO_GL=osmesa
SF=Experiment/Code/seeds_mission.json
L=Experiment/Code/out
RUN="python3 -u Experiment/Code/run_all_rqs.py --seeds_file $SF"

echo "PHASE1 35B-A3B parallel start $(date +%H:%M)" >> $L/master.log
CUDA_VISIBLE_DEVICES=0 $RUN --only 35B --shard 0/2 --no_evict >> $L/g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN --only 35B --shard 1/2 --no_evict >> $L/g1.log 2>&1 &
wait
echo "PHASE1 done $(date +%H:%M)" >> $L/master.log

# 35B evict → 디스크 확보 후 4B(GPU0) + 27B(GPU1) 병렬
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B
echo "PHASE2 4B+27B parallel start $(date +%H:%M)" >> $L/master.log
CUDA_VISIBLE_DEVICES=0 $RUN --only 4B  --no_evict >> $L/4b.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN --only 27B --no_evict >> $L/27b.log 2>&1 &
wait
echo "PHASE2 done $(date +%H:%M)" >> $L/master.log

python3 Experiment/Code/aggregate_rqs.py >> $L/aggregate.log 2>&1
echo "ALL DONE $(date +%H:%M)" >> $L/master.log
