# VLM-guided GCRL — 상차 미션 RQ 실험 스위트

로컬 **Qwen3.5**(HuggingFace, 네이티브 멀티모달)를 상위 플래너로 써서, 흙더미(pile)→트럭(truck)
상차 미션을 장애물 회피와 함께 수행하는 실험. 하위 제어기는 학습된 **GCRL(SAC+HER, v4base)**.

- **임무 지시**: "흙더미로 이동 후, 장애물을 회피하며 트럭으로 이동해라"
- **환경**: 흙더미 + 덤프트럭 + 랜덤 장애물(중앙 traverse 대). 씬은 seed로 결정론 재현.
- **A/B 대조**: A = 단독 GCRL이 [pile, truck] 직접 추종 / B = Qwen 서브골 시퀀스를 GCRL이 relay.
- **성공(safe_success)** = 흙더미 픽업점 도달(stage1) AND 트럭 투하점 도달(stage2) AND 충돌 0.
  (도달 허용오차 0.3 m — 컨트롤러 far-reach 정밀도 기준.)

## 연구 질문 (RQ)
| RQ | 질문 | 변수 | 고정 |
|---|---|---|---|
| **RQ1** | 멀티모달 입력의 영향 | RGB / height map / 둘다 | 27B, ID, full 프롬프트 |
| **RQ2** | 플래너 파라미터 수 | 4B → 27B → 35B-A3B → (122B) | both, ID, full |
| **RQ3** | OOD에서 VLM 서브골의 대처 | ID vs OOD × (A vs B) | 27B, both, full |
| **RQ4** | 프롬프트 엔지니어링 | full / −운동학 / −CoT / −clearance / minimal | 27B, both, ID |

- 100 seed(회피 필요 challenge 씬, `Experiment/Code/screen_mission_seeds.py`).
- RQ1/3/4는 크기 격리를 위해 **Qwen3.5-27B** 고정. RQ2만 사다리 스윕. 공유 셀(27B/both/ID/full)은 1회만 실행.

## 실행
```bash
# 1) 100 challenge seed 선별
python Experiment/Code/screen_mission_seeds.py --need 100
# 2) 프롬프트 변형 생성(RQ1/RQ4)
python Experiment/Code/make_prompts.py
# 3) 전체 RQ 셀 실행(재개 가능; 모델 그룹 단위 GPU/디스크 정리)
python Experiment/Code/run_all_rqs.py
# 4) figure/table/고찰 생성 → Experiment/RQ{1,2,3,4}/
python Experiment/Code/aggregate_rqs.py
```

## 산출물 (RQ별 폴더 `Experiment/RQ{1,2,3,4}/`)
- `figure.png` — 논문용 그림(막대/선, Wilson 95% CI)
- `table.csv`, `table.tex` — 논문용 표(booktabs)
- `discussion.md` — 결과 요약 + 고찰
- `data_*.csv` — 조건별 seed 단위 원자료

## 지표
safe_success(주지표), success, collision, stage1(pile 도달), stage2(truck 도달), path length,
avg subgoals. rate는 Wilson 95% CI.

## 코드 (`Experiment/Code/`)
- `qwen_planner.py` — Qwen 로컬 추론(모델-무관 Auto 클래스, thinking 파싱, RQ1 modality).
- `mission.py` — 미션 씬 생성/렌더(ID/OOD), GCRL 장애물 주입, pile→truck 2단계 롤아웃.
- `mission_run.py` — A/B 미션 러너(조건별).
- `run_all_rqs.py` — RQ 오케스트레이터(셀 dedup, 재개, 자원정리).
- `aggregate_rqs.py` — figure/table/고찰 생성.
- `make_prompts.py`, `screen_mission_seeds.py` — 프롬프트 변형 / seed 선별.
- `prompts/` — RQ1·RQ4 프롬프트 변형(system_prompt.txt 파생).
