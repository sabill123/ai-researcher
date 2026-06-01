# V1 시대 교훈 + v2 iter 0 postmortem (2026-04-21 갱신)

v1 era(447 실험, 2025-11 ~ 2026-04-17)에서 드러난 **구조적 실패 패턴**과
반복해선 안 될 기법들. v2 iter 0에서 추가로 확증된 실패 + 유효 기법도 포함.
새 실험을 설계할 때 이 문서를 먼저 참조하라.

---

## 0. 이중 목적 상기 (DUAL GOAL)

**모델의 목적은 두 가지이고 둘 다 동등 중요**:
- (A) 이미지 → severity 0~4 직접 예측 (MAE 지표)
- (B) 두 이미지 → 어느 쪽이 더 심한가 (Spearman |ρ| 지표, **pairwise ranking**)

(B)의 존재 이유: discrete 0~4 라벨은 정보 손실 많음. 환자 경과 추적·환자 간
비교 같은 실제 임상 use case는 ranking 판별력에 의존. **(A)만 좋아지면
임상 가치 없음.** 따라서 모든 proposal은:
- MAE 개선 hypothesis + |ρ| 개선(또는 최소 유지) hypothesis **둘 다** 제시
- Judge의 baseline 승격 조건: `MAE 개선 AND |ρ| 비퇴보`
- 한 지표만 개선되는 경우 → Judge가 "single-metric optimization" 경고, 다음 iter에서 보완 strategy 강제

---

## 1. 데이터 오염: annotator_auto는 절대 금지

**사실**: v1 training에 `annotator_auto` 4,977건이 섞여 있었음 (단일 최대
supervision). 이것은 외부 VLM이 아니라 **프로젝트 자체 FaRL 모델의 pseudo-label**
— self-distillation으로 모델 bias가 고착되어 true generalization이 왜곡됨.

**교훈**:
- 어떤 실험이든 training set에 `annotator_auto`, `auto_*`, `pseudo_*` 라벨이
  포함되면 **data_manifest gate가 dirty halt 해야 함**.
- 현재 v2는 `integrated_dataset_clean/`만 사용. annotator는 `annotator_01~04`
  + `라벨러1~4`만 허용. 다른 annotator 등장하면 즉시 의심.
- VLM pairwise supervision 도입 시에는 `annotator_auto`가 아니라 별도 BT loss
  track으로 명시 격리.

## 2. Rank-overfitting: MAE↓인데 |Spearman|↓ 이면 의심

**증거**: v1에서 MAE가 24% 개선되는 동안 |Spearman ρ|는 27% 악화함. 모델이
test 샘플의 severity 값에 over-fit하면서 **ordinal rank 학습은 퇴행**.
이 패턴은 clinical ordinal ranking 용도에서 MAE 숫자가 "가짜"임을 의미.

**교훈**:
- Judge sanity check 필수: `MAE↓` + `|ρ|↓` 동시 발생 시 무조건 fail.
- `BottleneckTracker.detect_rank_overfitting()`를 매 iteration 돌려
  parent의 |ρ| 대비 5% 이상 퇴보한 자식 experiment는 baseline 승격 금지.
- Proposal 설계 시 "MAE 개선" 뿐 아니라 "|ρ| 유지 또는 개선" 둘 다 가설에 포함.

## 3. 테스트셋 확장 아티팩트: 숫자 간 비교 금지

**증거**: 2026-04-08/09에 test set이 292 → 944 rows로 확장됨. 이 순간 MAE가
0.5496 → 0.4190으로 "개선"된 것처럼 보였으나, 원본 292 set에서 공정 재평가
결과 exp_579는 실제로 **MAE 0.6831**이었음 (기록상 best exp_203).

**교훈**:
- **다른 test set·era 간 숫자 비교 절대 금지**. clean_vanilla_mse_wk=0.4088은
  948-row BT test set 기준. v1의 0.5496과 *같은 지표가 아님*.
- 새 baseline 승격은 반드시 **동일 test set 해시** 기준 improvement 체크.
- `experiment_version` 필드를 분리해 v1 exps가 v2 parent selection에 섞이지
  않도록 이미 차단 (DEFAULT_VERSION_FILTER="v2").

## 4. 병목 action은 이마_주름에서 입_우로 이동

**v1**: 이마_주름 MAE=0.7940, gap_share=**92%**. 이 단일 action이 전체 실패를
좌우했으나 시스템이 집중 공략하지 못함.

**v2 현재 (clean_vanilla_mse_wk 기준)**: 
- 입_우 MAE=0.541 (gap_share 43%)
- 안면_무표정 0.450, 이마_주름 0.441, 입_이 0.421, 눈_질끈 0.296, 눈_살짝 0.305

**교훈**:
- clean 데이터 재학습 후 병목이 **이마_주름에서 입_우로 변경됨**. 
  v1 시기 이마_주름 집중 공략 기법(geometric_aug, forehead_roi 등)은
  v2에선 **ROI가 제한적일 가능성** — 입_우를 우선 타겟.
- 이마_주름은 인간 annotator 간 일치도도 55% 수준이라 task 자체 한계가 있음
  (VLM 평가에서도 이마_주름 56~66%). 무리한 개선 시도보다 다른 action 우선.
- 단 한 action만 90%+ 병목 재발생시 데이터 distribution 재확인 우선.

## 5. 실패 확증된 v1 기법 (반복 금지)

이 기법들은 v1에서 **반복적으로 퇴보**를 보였으므로 재제안 금지 또는 엄격한
정당화 필요:

- **단순 augmentation 강화** (rotate/flip/color jitter alone): 눈 action은
  improve 가능하나 이마_주름엔 무의미. v1에서 8회 시도, 전부 regression.
- **Backbone swap without head change**: FaRL → other 단독 교체는 ~0.02 MAE 
  감소뿐, Spearman regression 동반. 
- **MSE 가중치 tuning만**: loss:action_weights 단일 시도 6회, 평균 delta +0.008
  (퇴보).
- **Temperature scaling post-hoc**: 학습 안 바뀌면 MAE 거의 불변.
- **Ensemble averaging of bad models**: 각 시드 성능이 나빠진 상태에서
  ensemble해도 MAE 개선 안 됨 (v1 exp_525 clone 실패 사례).

### v2 iter 0에서 추가 확증 (2026-04-21)
- **CORN/CORAL ordinal head 단독**: `exp_003_ipi_corn_ordinal_head` 0.4748 (+0.066).
  architecture 단일 변경만으로는 baseline 돌파 불가. CORN은 **다른 기법과 조합
  시에만** 의미.
- **Neutral-face contrastive loss 단독**: `exp_002_neutral_contrastive_anchor_head`
  0.4863 (+0.077). loss 추가 단독은 `v1 §5 단순 augmentation`와 같은 패턴.
- **VLM weight 단독 축소**: `v2_exp_002` 0.4877 (+0.079), parent와 flat. 
  **VLM weight는 주 lever 아님**. BT weight가 실질 lever.
- **Pure BT loss (severity 제거)**: `clean_pairwise_bt_v2` 0.5549 (+0.146).
  severity supervision 없이 BT만으로는 정보 부족.

### ⚠️ iter 2+ 절대 금지 기법 (하드 차단)

**다음 기법들은 v2 era 8+ 시도 모두 baseline 대비 2σ 이상 퇴보 확증. Engineer는 proposal에 아래 키워드 중 하나라도 포함하면 Judge가 VETO 행사 의무:**

- `SORD` (Soft Ordinal Regression Distribution)
- `marginrank` / `margin_ranking`
- `RankNet` / `ranknet`
- `SoDeep` / `sodeep`
- `ranksim` / `rank_surrogate`
- `NeuralNDCG`
- `Fast-Soft-Sort` / `fastsoftsort`
- `Plackett-Luce` / `plackett_luce`
- `Kendall τ surrogate`
- `Rank-N-Contrast` / `rank_n_contrast`
- `pure BT loss` (severity 없이 BT만)
- `BT + VLM weight` 변형 (어떤 weight 조합도)
- `CORN/CORAL ordinal head` (단독 또는 rank loss와 조합)
- `Uncertainty-weighted multi-loss` (exp_579 재현)
- `Neutral-face contrastive` (exp_002 계열)

**모두 공통 실패 메커니즘**: severity 학습을 방해하여 MAE 악화. v2 era에서 9+ 실험으로 확증.

**허용 방향** (paradigm shift):
- **Backbone 교체**: FaRL → DINOv2 / SAM / CLIP / VideoMAE
- **Self-supervised pretraining** on unlabeled 안면마비 데이터
- **Multi-task with Facial Action Units (FAU)**
- **Cross-attention 기반 쌍대비교 직접 모델링**
- **Curriculum learning** (severity 0·4 극단부터 → 중간)
- **Teacher-student distillation**
- **데이터 증강 방향**: VLM pairs 2-model agreement로 완화하여 5,000+ 확장
- **Patient-level split** 별도 벤치마크

### v2 iter 1에서 추가 확증 (2026-04-21 저녁, 3-seed σ=0.0066 기반)

**핵심 교훈**: **Rank-aware loss 계열(SoDeep, RankNet, CORN stacking, Plackett-Luce) 
전원 실패 — baseline 대비 +0.09~+0.12 심각한 퇴보**. 

- **SoDeep differentiable Spearman + BT + continuous score** (exp_004): 0.504+, **+0.095**
- **RankNet + CORN ordinal + BT aux** (exp_005): 0.527+, **+0.118**
- **Fast-Soft-Sort + BT + Uncertainty** (exp_006): crash (`KeyError: pairwise_ranking`)
- **Plackett-Luce listwise** (exp_007): Engineer 구현 단계 실패

**확정된 결론**:
1. **Rank-aware loss 추가는 severity 학습을 방해**. v1에서 의심됐던 패턴 v2에서 재확인.
2. **Vanilla MSE+WK가 실제로 매우 강한 baseline**. 순수 severity 학습만으로도
   |ρ|=0.457을 달성. 여기에 BT/rank 추가해도 |ρ| 개선이 MAE 악화를 보상 못 함.
3. **이 방향 (rank-aware loss 추가)은 dead-end**. 새 iter에서 동일 계열 재제안 금지.

### v2 iter 1 크래시 패턴 (Engineer 구현 실패)
- `KeyError: 'pairwise_ranking'`: Engineer가 train loop에서 test-only dataset
  key를 접근. `dataset.py`는 `mode='test'`에만 `pairwise_ranking` 추가하는데
  Engineer가 이를 간과.
- Claude CLI `exit 1`: 복잡한 구현 요청 시 LLM 응답 실패.

**교훈**: Engineer proposal에 **rank-loss 계열 2개 이상 stacking**은 구현 난이도가
sandbox 검증 수준을 넘어서 crash 확률 높음. 단일 rank loss + 기존 severity 유지
조합이 더 안전.

## 6. v1에서 유일하게 일관된 이득을 준 것

- **Landmark-guided spatial mask** (exp_203 계열): 얼굴 영역별 attention.
  exp_203이 v1 공정 재평가 best.
- **Action-specific head split**: `model_type=separate`가 joint보다 일관 우위.
- **MSE+WK loss**: WeightedKappa가 severity ordinal 특성을 잘 담음. v2
  baseline도 이것 채택.
- **Uncertainty-weighted multi-loss (exp_579 계열)**: 개선 있음, 단 rank
  regression 주의.

### v2 iter 0에서 발굴한 유망 방향 (재활용 권장)
- **입_우 side-aware landmark augmentation** (`exp_001_ipwoo_sideaware_noflip_asymlmk`,
  0.4385): iter 0 **최선** 결과. `noflip + 좌우 비대칭 landmark delta`가 주효. 
  - **bilateral asymmetry는 입_우의 본질** (한쪽 mouth pull). flip augmentation이 
    오히려 signal을 파괴. 
  - 이 개념을 **입_이, 안면_무표정에도 확장** 권장 (bilateral 해부학적 특성).
- **BT weight 축소** (`v2_exp_003` bt=0.2, 0.4593): parent(`v2_exp_001` bt=0.3, 0.4789) 
  대비 −0.020. 추가로 **bt=0.1, 0.05** 탐색 가치. 단 rank 학습 보강과 병행해야 함.

## 7. Tree 구조 원칙 (v1 실패 답습 방지)

**v1 문제**: 447 exps 중 136 leaf, 44 orphan root. `exp_045` 하나가 118 노드
subtree(73%) 독점. 나머지 42 root는 대부분 단발 시도. **개선 24 vs 퇴보 93**
— 깊이 들어가기만 했지 수렴 없음.

**v2 강제 규칙**:
- 모든 새 proposal은 **기존 baseline의 자식**으로 등록 (parent_id 필수).
- 같은 iteration에서 같은 parent 2회 이상 재사용 금지 (diversity).
- baseline 승격은 MAE + |ρ| + sanity 3중 체크 통과 시에만.
- 연속 6회 미개선 → `stagnation_engine`이 강제 paradigm shift 트리거.

## 8. 반드시 고려할 가설 설계 체크리스트

새 proposal을 제출하기 전 Engineer가 스스로 체크:

- [ ] 이 기법은 현재 primary bottleneck action을 실제로 겨냥하는가?
- [ ] v1에서 유사 기법이 몇 번 시도됐고 결과는? (operational_memory 조회)
- [ ] MAE 개선 가설과 함께 **|ρ| 유지/개선** 메커니즘을 설명했는가?
- [ ] parent experiment의 per-action profile과 비교했을 때 어떤 action이
      개선되고 어떤 action은 regress할 위험이 있는가?
- [ ] 구현이 400라인 미만인가? (sandbox 차단 방지)
- [ ] read-only 파일을 건드리지 않는가? (utils.py, backbone.py, pairwise_dataset.py)
- [ ] 구현이 기법 **단일 변경**이 아니라 **2~3 기법 조합** stacking 인가?
      (v2 iter 0에서 단일 변경 전원 실패 — §5 참조)

## 9. Iter 1+ 전략 가이드 (iter 0 결과 기반)

**핵심 발견**: iter 0의 4 proposal 중 3 완주 결과가 모두 baseline 대비 +0.030~+0.079
MAE 퇴보, 어떤 것도 통계적으로 유의하지 않음 (CI width 0.165). **단일 변경 기법의
한계**가 확인됨. 동시에 입_우 augmentation(exp_001)과 BT weight 축소가 부분 signal.

### 9.1 Iter 1 이상 권장 전략
1. **Stacking 필수**: augment + arch + loss 중 **최소 2개 조합**. iter 0 단일 변경은 전원 noise 수준.
2. **입_우 중심 + bilateral 확장**: exp_001의 side-aware 개념을 다른 bilateral action (입_이, 안면_무표정)으로 이식. Action-specific sideaware head + 좌우 비대칭 augmentation.
3. **Rank 학습 보강**: |ρ| 현재 0.457은 MAE보다 **더 큰 gap** (−35% vs −30%). 단순 BT loss로는 부족 확인됨. 
   - 명시적 **ranking loss** (margin ranking, Spearman soft) 도입
   - Continuous score head + ranking auxiliary
   - BT weight 0.1 이하 + rank loss 0.3~0.5 권장 조합
4. **이마_주름 지양**: 인간 일치도 55%로 task 상한이 낮음. 무리한 공격보다 입_우·입_이 개선에 자원 집중.
5. **Sanity check 자체가 실험 방향**: Judge가 rank_regression_avoided=False 반복 감지 → 다음 iter는 **rank 중심 loss 조합** 우선 제안.

### 9.2 Sanity fail 시 Judge 동작 (자율성 원칙)
- `sanity_checks_passed=False`여도 **Judge는 iter 중단하지 않음**. 대신 다음 iter의 
  `engineer_directive`에 명확한 교훈 주입:
  - "iter N-1의 4 proposal 모두 rank regression, 단일 변경. iter N은 반드시 
    rank-aware + multi-stack"
  - `forbidden_techniques`에 단일 기법 형태 재제안 자동 차단
- 사람 승인 없이 ANNA가 축적된 knowledge로 자가수정 진행.
- orchestrator는 sanity fail을 **warning으로만 기록**하고 Phase 3~9 계속 진행.

---

## 참조

이 문서는 메모리 디렉토리의 다음 파일들에서 증류:
- khu_anna_overfitting_evidence.md, khu_anna_overfitting_detector.md
- khu_mae_comparability_2026-04.md, khu_marker_reeval_2026-04-19.md
- khu_auto_labeling_contamination.md, khu_dataset_rules.md
- khu_known_pitfalls.md, khu_v2_audit_2026-04-20.md
- 2026-04-21 갱신: ANNA v2 iter 0 postmortem (exp_001~004 결과)
