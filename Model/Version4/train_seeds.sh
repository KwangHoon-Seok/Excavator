#!/bin/bash
# 채택 GCRL config(start-fixed × dist)를 seed 1,2로 2M 학습(3-seed 분산용). GPU 1개씩 병렬.
cd /home/elicer/Excavator
export MUJOCO_GL=osmesa
D=Model/Version4
echo "SEED-TRAIN START $(date +%m-%d_%H:%M)" > $D/train_seeds.log
CUDA_VISIBLE_DEVICES=0 python3 -u $D/train.py $D/config_adopted_s1.yaml --steps 2000000 > $D/train_adopted_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -u $D/train.py $D/config_adopted_s2.yaml --steps 2000000 > $D/train_adopted_s2.log 2>&1 &
wait
echo "SEED-TRAIN DONE $(date +%m-%d_%H:%M)" >> $D/train_seeds.log
# 완료 후 3-seed mean±std 그림 생성
python3 Experiment/Code/gcrl_final_fig_3seed.py >> $D/train_seeds.log 2>&1
