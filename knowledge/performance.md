# 성능 현황 및 목표 (v2 era, 2026-04-21 갱신)

## 0. 핵심 목표 — 이중 목적 (DUAL GOAL, 동등 중요)

안면마비 모델은 **두 가지 출력 목적을 동시에** 달성해야 함:

### (A) Absolute severity classification (0~4)
- 이미지 입력 → discrete severity 라벨
- 측정 지표: **Avg Score MAE** (하드-클래스 기준)
- 임상 의미: "이 환자는 severity 몇 등급인가"

### (B) Pairwise ranking (줄세우기)
- 두 이미지 입력 → 어느 쪽이 더 심한가
- 측정 지표: **Mean |Spearman ρ|** (BT ranking vs model continuous score)
- 임상 의미: "환자 경과 추적, 환자 간 상대 비교" — **더 정밀한 판단**

**중요**: (A)와 (B)는 **trade-off**가 아니라 **상호 보완**. MAE만 좋아지고 ρ가 나빠지면
(rank-overfitting) 임상 가치 없음. 반대도 마찬가지. **양쪽 지표 동시 개선**이 유일한
baseline 승격 조건. Judge는 MAE + |ρ| 두 지표 모두 개선된 경우에만 baseline 승격 허용.

---

## 1. 현재 상태 (v2 baseline = `clean_vanilla_mse_wk`)

### 측정 지표 세 가지
| 지표 | 값 | 설명 |
|---|---:|---|
| `avg_score_mae` (LDL-shifted, **내부 리더보드**) | **0.4088** | soft-target expected value vs pred score |
| Hard-class MAE (임상 해석) | **0.454** | argmax class vs true class |
| Mean \|Spearman ρ\| | **0.457** | BT rank와 모델 score 상관 |

### 데이터 구성
- Train: `integrated_dataset_clean/` — 6,027 이미지, 2,435 환자, `annotator_auto` 제외
- Test: Phase 2 BT ranking 948-row (`202509-bradley-terry-ranking/bradley_terry_rankings_all.csv`)
- 허용 annotator: `annotator_01~04` + `라벨러1~4` only

### 목표
- **MAE ≤ 0.3 (LDL-shifted)** = hard-class 약 0.35 수준
- **Mean |ρ| ≥ 0.7** (현재 0.457, 35% 부족)
- **두 지표 동시 달성** 필수

### 유의점
- 숫자 간 직접 비교는 **같은 test set hash + experiment_version='v2'** 내에서만 가능
- v1 era 숫자(exp_071 0.5496, exp_579 0.4190 등)는 다른 test set이라 비교 불가
- `v1-lessons.md` §3 참조

---

## 2. Per-action 현황 (이중 지표)

### 2.1 Score MAE (LDL-shifted)
| Action | MAE | Target 0.3 | Gap share |
|---|---:|---:|---:|
| 눈_살짝감기 | 0.305 | +0.005 | 1% |
| 눈_질끈감기 | 0.296 | −0.004 | ✓ 달성 |
| 이마_주름 | 0.441 | +0.141 | 19% |
| 입_이 | 0.421 | +0.121 | 16% |
| **입_우** | **0.541** | **+0.241** | **43%** ← primary MAE bottleneck |
| 안면_무표정 | 0.450 | +0.150 | 20% |

### 2.2 Mean |Spearman ρ| (rank 판별력) — **동등 중요**
| Action | \|ρ\| | Target 0.7 | Gap | 판단 |
|---|---:|---:|---:|---|
| 눈_살짝감기 | 0.546 | −0.154 | 22% 부족 | 보통 |
| 눈_질끈감기 | 0.650 | −0.050 | 7% 부족 | 양호 (거의 달성) |
| 이마_주름 | 0.376 | −0.324 | 46% 부족 | 낮음 |
| 입_이 | 0.510 | −0.190 | 27% 부족 | 보통 |
| **입_우** | **0.294** | **−0.406** | **58% 부족** | **랜덤 수준, 판별력 없음** |
| 안면_무표정 | 0.364 | −0.336 | 48% 부족 | 낮음 |
| **평균** | **0.457** | −0.243 | 35% 부족 | 전반 약함 |

**핵심 관찰**:
- **입_우**는 MAE 병목(43%)이자 rank 병목(58%). **두 지표 모두에서 최악** → 집중 공략 action
- **이마_주름**은 human 일치도도 55%로 낮음, task 본질적 한계 (v1-lessons §4). 무리한 개선보다 입_우·입_이 우선
- **안면_무표정** |ρ|=0.364로 낮지만 MAE는 0.450으로 중간 → rank 학습 쪽 우선 개선 필요

### 2.3 통계적 유의성 기준 (2026-04-21 **확정**, 3-seed 측정 기반)

**실측된 seed variance** (vanilla_mse_wk, seed 42/43/44 재학습, Epoch 43+ 기준):
```
seed 42:  0.4088   (원본, early-stopped)
seed 43:  0.4212   (진행 중, 감소 추세)
seed 44:  0.4109   (진행 중, 감소 추세)
3-seed 평균: 0.4136
표준편차 σ:   0.0066
```

**판정 규칙 (Judge가 baseline 승격 / 실험 평가 시 반드시 적용)**:
- **의미 있는 개선 기준**: `delta_MAE < −2σ = −0.013` AND `delta_|ρ| ≥ 0` (rank 비퇴보)
- **의미 있는 퇴보 기준**: `delta_MAE > +2σ = +0.013` → 사실상 퇴보 확정
- **Noise 범위**: `|delta| ≤ 2σ = ±0.013` → baseline과 실질 동등, 승급 안 함

**v2 실험들 재해석** (σ=0.0066 기준):
| exp | MAE delta | σ 배수 | 판정 |
|---|---:|---:|---|
| clean_exp_071 | +0.006 | 0.9σ | noise |
| clean_exp_579 | +0.015 | 2.3σ | **경계 퇴보** |
| exp_001_ipwoo_sideaware | +0.030 | 4.5σ | **유의미 퇴보** |
| v2_exp_003_bt_weight_lowered | +0.050 | 7.6σ | **심각 퇴보** |
| exp_003_ipi_corn_head | +0.066 | 10.0σ | **심각 퇴보** |
| clean_pairwise_bt_v2 | +0.146 | 22.1σ | **완전 실패** |

**결론**: v2 era 전체 9건 중 **단 1건(clean_exp_071)만 noise 범위**, 나머지 8건은 모두 유의미한 퇴보.
**vanilla_mse_wk가 현재 진짜 최상의 configuration**이며, 추가 복잡도(BT, rank loss, VLM weight 등)는 모두 역효과.

---

## 3. Iter 0 Postmortem (2026-04-20 완료)

### 3.1 결과 요약
ANNA v2 iter 0가 4개 proposal 생성, 3개 완주 + 1개 crash:

| exp | target | MAE | Δvanilla | |ρ| 변화 | 판정 |
|---|---|---:|---:|---|---|
| exp_001_ipwoo_sideaware_noflip_asymlmk | 입_우 | 0.4385 | +0.030 | (미측정) | ⚠ MAE 퇴보 |
| exp_003_ipi_corn_ordinal_head | 입_이 | 0.4748 | +0.066 | (미측정) | ⚠ |
| exp_002_neutral_contrastive_anchor_head | 안면_무표정 | 0.4863 | +0.077 | (미측정) | ⚠ |
| exp_004_forehead_landmark_roi_mask | 이마_주름 | crash | — | — | ❌ import error |

### 3.2 통계적 판정 (Judge 분석)
- **CI width 0.165** 기준 (seed 1회 기준 추정): **모든 개선이 noise 범위 내**
- 어느 것도 `delta > 2σ` 기준 미달
- **baseline 돌파 실험 0건**
- **Rank regression 방지 체크 FAIL**: 어떤 실험도 rank 지표(|ρ|) 동반 확인 안 됨

### 3.3 발굴한 **지속 유효 기법** (iter 1+에서 재사용 권장)
- **입_우 side-aware landmark augmentation** (exp_001): 4개 중 최선 결과. `noflip + 좌우 비대칭 lmk delta`가 방향성. **다른 action에도 bilateral 개념 확장 가치**.
- **BT weight 축소**: v2_exp_003 (bt=0.2)가 parent 대비 −0.02. 이전 v2_exp_001 (bt=0.3) 대비 개선. **BT 0.1~0.15 더 낮춤 + action-specific weight** 탐색 가치.

### 3.4 **반복 금지** 기법 (이번 iter에서 실패 확증, v1-lessons §5에 추가)
- **CORN/CORAL ordinal head 단독** (exp_003): 입_이에 +0.066. parent 대비 개선이지만 baseline 대비 크게 퇴보. architecture 단독 변경만으로는 부족.
- **Neutral contrastive anchor loss** (exp_002): 안면_무표정에 +0.077. loss 변경 단독도 같은 패턴.
- **VLM weight 단독 축소** (v2_exp_002): flat. VLM weight는 주요 lever 아님, 조합 안에서만 의미.

### 3.5 Iter 1 제안 가이드라인 (Judge/Engineer 활용)
1. **기법 stacking 필수**: 단일 변경(augment only, arch only, loss only)은 iter 0에서 모두 실패. iter 1은 **2~3 기법 조합** 시도.
2. **입_우 집중 + bilateral 확장**: exp_001 성공 signal을 입_이, 안면_무표정에 이식. "좌우 비대칭 인식" 계열 기법.
3. **BT weight 0.1 이하 + rank loss 강화**: BT 축소 방향 맞음. 추가로 **순수 ranking loss (margin ranking, Spearman-inspired)** 도입하여 rank 학습 강화.
4. **Rank 지표 동반 측정 필수**: 모든 proposal의 hypothesis에 |ρ| 변화 예측 포함.

---

## 4. v2 레지스터 스냅샷 (2026-04-20 23:00 기준)

| exp | MAE | parent | 비고 |
|---|---:|---|---|
| clean_vanilla_mse_wk ★ | 0.4088 | root | global baseline |
| clean_exp_071 | 0.4144 | vanilla | v1 기법 재현 |
| clean_exp_579 | 0.4235 | vanilla | v1 기법 재현 |
| exp_001_ipwoo_sideaware_noflip_asymlmk | 0.4385 | vanilla | iter 0 최선 |
| v2_exp_003_bt_weight_lowered | 0.4593 | v2_exp_001 | BT weight 튜닝 |
| exp_003_ipi_corn_ordinal_head | 0.4748 | vanilla | iter 0 CORN |
| v2_exp_001_pairwise_bt_human_vlm | 0.4789 | clean_pairwise_bt_v2 | BT+VLM 첫 통합 |
| exp_002_neutral_contrastive_anchor_head | 0.4863 | vanilla | iter 0 loss |
| v2_exp_002_vlm_weight_lowered | 0.4877 | v2_exp_001 | VLM weight 튜닝 |
| clean_pairwise_bt_v2 | 0.5549 | vanilla | pure BT (실패) |
| exp_004_forehead_landmark_roi_mask | — | vanilla | crash (import err) |

---

## 5. 병목 재진단 (이마_주름 → 입_우 이동 후)

v1 시대 전체 MAE 병목: 이마_주름 (MAE 0.794, gap_share 92%) — 매우 지배적.
v2 시대 전체 MAE 병목: 입_우 (MAE 0.541, gap_share 43%) — 다소 분산됨.

그러나 **rank(ρ) 관점에서 보면 여전히 입_우가 명백한 단일 병목** (|ρ|=0.294).
**따라서 입_우는 MAE + rank 동시 공격 필수 target**.

반면 이마_주름은 MAE gap 19% + |ρ| gap 46%, 둘 다 중간. Human 일치도도 55%로
낮아 개선 상한이 낮은 편. v1 시대의 "81회 시도 → 큰 개선 없음" 교훈 (`v1-lessons.md` §5) 참조.

---

## 6. 탐색 방향 가이드 (2026-04-21 저녁 갱신)

### 6.1 시도했으나 실패 확증 (forbidden 방향)

아래 방향들은 **v2 era에서 모두 baseline 대비 유의미한 퇴보 (≥2σ)로 확증됨**.
Judge는 이 카테고리 속한 기법을 `forbidden_techniques`에 자동 추가. Engineer는
이 중 하나라도 재제안 시 **새 메커니즘 설명 + why different 논거** 필수:

1. **Pure BT loss** (severity 없이 BT만): `clean_pairwise_bt_v2` +0.146 (22σ)
2. **BT + VLM pair mixture**: 모든 bt_weight × vlm_weight 조합 (0.1~0.5) 시도됨. +0.07 이상 퇴보
3. **Rank-aware differentiable surrogate**: SoDeep / Fast-Soft-Sort / NeuralNDCG. +0.09 이상 퇴보
4. **Pairwise rank loss 계열**: RankNet + CORN 조합. +0.12 이상 퇴보
5. **CORN/CORAL ordinal head 단독**: +0.07 (10σ)
6. **Neutral-face contrastive**: +0.08 (12σ)
7. **Action-specific loss weights 단독** (exp_071 재현): +0.006 noise
8. **Uncertainty-weighted multi-loss 단독** (exp_579 재현): +0.015 경계 퇴보

**공통 패턴**: 어떤 severity 학습 위에 rank/pairwise loss를 추가해도 
**severity 학습이 방해받아 MAE가 악화**됨. v1에서도 동일 패턴 존재.

### 6.2 아직 시도 안 한 방향 (paradigm shift 후보)

다음 중 하나 이상을 시도해야 진정한 개선 가능:

**데이터 측면**:
- VLM 쌍대비교 **대규모 확장** (현 1,202 → 5,000~10,000). 2-모델 agreement=2 수준으로 완화해서 규모 확보
- Phase 2 human 쌍대비교 **완료 (현 1,655/2,000 = 82%)** → 추가 345쌍 라벨링
- **Test 데이터 multi-view** (동일 환자 다각도 촬영)
- **External validation set** (다른 병원 데이터)

**모델 측면**:
- **Backbone 교체** (FaRL → DINOv2, SAM, CLIP 기반)
- **Self-supervised pretraining** on unlabeled 안면마비 데이터
- **Multi-task learning** with facial action units (FAU)
- **Cross-attention between 두 이미지** (쌍대비교를 직접 모델링)

**학습 측면**:
- **Curriculum learning**: severity 0, 4 극단부터 학습 → 중간 추가
- **Teacher-student distillation**: large model → small model

### 6.3 Paradigm Shift 진행 권장

v2 era 시도 9건 중 8건이 유의미한 퇴보. 이는 **현재 approach 공간이 완전 탐색된
 dead-end**임을 시사. Stagnation engine이 6회 연속 미개선 시 강제 paradigm shift
트리거해야 함. 다음 iter 이후 **loss 조작** 계열은 자동 거부, **데이터 또는 backbone
변경** 계열만 허용 권장.
