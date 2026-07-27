#!/bin/bash
# 1M 4버전 학습 완료 대기 → 최고 safe_success 승자 선정 → 2M까지 이어학습(seed 3).
cd /home/elicer/Excavator
export MUJOCO_GL=osmesa
D=Model/Version4
LOG=$D/chain_2M.log
echo "CHAIN 대기 시작 $(date +%H:%M)" > $LOG

# 1) 1M 학습 4개 모두 끝날 때까지 대기 (train.py 프로세스 소멸)
while pgrep -f 'config_gcrl_v[0-9]_' >/dev/null 2>&1; do sleep 60; done
sleep 20
echo "1M 학습 완료 감지 $(date +%H:%M)" >> $LOG

# 2) 승자 선정
W=$(python3 Experiment/Code/pick_winner.py)
echo "승자 = $W" >> $LOG

# 3) 1M 산출물 백업(곡선·체크포인트 보존)
CK=Model/checkpoints/GCRL_compare
cp $CK/$W.zip $CK/${W}_1M.zip 2>/dev/null
cp Model/logs/GCRL_compare/$W/fulltask.csv Model/logs/GCRL_compare/$W/fulltask_1M.csv 2>/dev/null
cp Model/logs/GCRL_compare/$W/metrics.csv  Model/logs/GCRL_compare/$W/metrics_1M.csv  2>/dev/null

# 4) 승자 2M 이어학습 (replay buffer 복원 → 1M→2M, seed 3은 config에 포함)
echo "2M 이어학습 시작 $(date +%H:%M): $W" >> $LOG
python3 -u $D/train.py $D/config_$W.yaml --resume $CK/$W.zip --steps 2000000 >> $D/train_${W}_2M.log 2>&1 &
TPID=$!
# 2M 동안 실시간 성능/curriculum 저장 계속
while kill -0 $TPID 2>/dev/null; do python3 Experiment/Code/gcrl_compare.py >/dev/null 2>&1; sleep 180; done
python3 Experiment/Code/gcrl_compare.py >/dev/null 2>&1
echo "2M 완료 $(date +%H:%M): $W → 최종 모델 $CK/$W.zip" >> $LOG
