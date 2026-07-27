# GCRL 2×2 학습 실시간 비교

최신 스냅샷 (fulltask 평가, n_eval_full=50):

| version | step | success | safe_success | collision |
|---|---|---|---|---|
| v1 startX·dist | 3000000 | 0.660 | 0.600 | 0.060 |
| v2 startO·dist | 1000000 | 0.440 | 0.280 | 0.280 |
| v3 startX·auto | 1000000 | 0.600 | 0.440 | 0.180 |
| v4 startO·auto | 1000000 | 0.380 | 0.280 | 0.160 |

**현재 선두**: v1 startX·dist (safe_success=0.600 @ 3000000 step)

※ 학습 진행 중 자동 갱신. 곡선은 progress.png.
