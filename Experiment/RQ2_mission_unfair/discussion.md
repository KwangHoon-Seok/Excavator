# RQ2 — Does the VLM mitigate the long-horizon limitations of GCRL?

**설정**: ID 환경(고정 레이아웃), n=100 seeds. 저수준 컨트롤러 = adopted GCRL(start-fixed × dist),
플래너 = Qwen3.5-27B. A = 단독 GCRL이 [흙더미, 트럭] 직접 추종 / B = VLM 서브골 relay.

## 결과 (success rate)
| metric | Single GCRL (A) | VLM+GCRL (B) | Δ |
|---|---|---|---|
| Success | 0.000 | 0.990 | +0.990 |
| Safe-success | 0.000 | 0.460 | +0.460 |
| Pile reach (stage1) | 1.000 | 1.000 | +0.000 |
| Truck reach (stage2) | 0.000 | 0.990 | +0.990 |
| Collision | 0.310 | 0.540 | +0.230 |

## 고찰
- **단독 GCRL success 0.00 ≪ VLM+GCRL 0.99** — VLM 서브골이 long-horizon 임무 성공률을 크게 끌어올림.
- 단계 분해가 핵심 증거: 단독 GCRL은 **흙더미 도달 1.00** 은 되지만 **트럭 도달 0.00** 에서 막힘.
  즉 컨트롤러가 흙더미(중간 swing)엔 가지만 흙더미→트럭의 **큰 swing(far goal)** 을 한 번에 수행 못 함.
- VLM은 이 long-horizon 문제를 **짧은 서브골 hop**으로 쪼개어(trucك 도달 0.99), GCRL이 각 hop을
  순차 추종하도록 만들어 임무를 완수시킴. → "VLM sub-goal 생성이 long-horizon 문제를 잘게 쪼개 해결 가능한
  문제로 바꿔준다"는 주장을 정량적으로 뒷받침.
- 평가를 OOD가 아닌 **ID에서** 수행해, 분포 변화가 아니라 순수 long-horizon 능력만 격리해 측정.
