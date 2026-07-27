#!/bin/bash
# GCRL 4버전(2×2: random_start × curriculum) 병렬 학습, 각 1M steps.
# GPU0: v1,v2(dist) / GPU1: v3,v4(auto). 로그는 train_vN.log, 체크포인트는 checkpoints/GCRL_compare/.
cd /home/elicer/Excavator
export MUJOCO_GL=osmesa
D=Model/Version4
STEPS=1000000
echo "TRAIN START $(date +%H:%M)" > $D/train_master.log

CUDA_VISIBLE_DEVICES=0 python3 -u $D/train.py $D/config_gcrl_v1_startX_dist.yaml --steps $STEPS > $D/train_v1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 -u $D/train.py $D/config_gcrl_v2_startO_dist.yaml --steps $STEPS > $D/train_v2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u $D/train.py $D/config_gcrl_v3_startX_auto.yaml --steps $STEPS > $D/train_v3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u $D/train.py $D/config_gcrl_v4_startO_auto.yaml --steps $STEPS > $D/train_v4.log 2>&1 &
wait
echo "TRAIN DONE $(date +%H:%M)" >> $D/train_master.log
