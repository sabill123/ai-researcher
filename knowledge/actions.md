# 안면마비 중증도 예측 — Action 정의

## 6개 TARGET ACTIONS

| 액션 | SB 매핑 | 신경 분지 | 선정 근거 |
|------|---------|-----------|-----------|
| 눈_살짝감기 | Eye closure (E) | Zygomatic | 각막 보호, SB 필수 |
| 눈_질끈감기 | Eye closure (E) | Zygomatic | 최대 노력 시 눈 폐쇄 능력 |
| 이마_주름 | Forehead wrinkle (F) | Temporal | 상부/하부 안면신경 감별 핵심 |
| 입_이 | Open mouth smile (M) | Buccal | 사회적 기능 핵심, SB 필수 |
| 입_우 | Lip pucker (P) | Marginal mandibular | 구륜근 기능, 유일한 해당 분지 평가 |
| 안면_무표정 | Resting symmetry | - | 안정 시 좌우 대칭 평가 |

## 중증도 기준 (Modified Sunnybrook Scale)

- Severity 0 (Grade 5): 정상 — 완전한 대칭
- Severity 1 (Grade 4): 경미한 약화 — 미세한 비대칭
- Severity 2 (Grade 3): 중등도 약화 — 뚜렷한 비대칭
- Severity 3 (Grade 2): 심한 약화 — 심각한 비대칭
- Severity 4 (Grade 1): 완전 마비 — 움직임 없음

## 액션별 평가 기준

### 이마_주름

- Grade 4 (정상): 눈썹 수평선 대칭
- Grade 3: 이마주름 비대칭
- Grade 2: 눈썹 비대칭 (수평선 걸침)
- Grade 1 (완전 마비): 이마주름 없음, 눈꺼풀 처짐
- **난이도 특성**: 미묘한 이마 주름 차이 → multi-scale feature 또는 high-res crop이 효과적
- **Class 분포**: Class 2(중등도)가 65.8% 지배적 → class-balanced loss 필수

### 눈_살짝감기

- Grade 4 (정상): 완전 폐쇄
- Grade 3: 눈꺼풀만 비대칭
- Grade 2: 동공 안 보임 수준
- Grade 1 (완전 마비): 동공 보임 수준
- **난이도 특성**: 상대적 양호. eye region의 미세한 closure 정도 차이

### 눈_질끈감기

- Grade 4 (정상): 완전 폐쇄
- Grade 3: 눈꺼풀만 비대칭
- Grade 2: 동공 안 보임 수준
- Grade 1 (완전 마비): 동공 보임 수준
- **난이도 특성**: 최대 노력 조건이라 차이가 눈_살짝감기보다 명확
- **Class 분포**: Class 2가 75.6% 지배적

### 입_이

- Grade 4 (정상): 정중선 대칭
- Grade 3: 입주변 주름 비대칭
- Grade 2: 입모양/팔자주름 비대칭
- Grade 1 (완전 마비): 정중선 완전 벗어남
- **난이도 특성**: 미소 시 비대칭 정도. 입 영역 ROI crop이 도움

### 입_우

- Grade 4 (정상): 정중선/입모양 대칭
- Grade 3: 입주변 주름 비대칭
- Grade 2: 입모양 비대칭
- Grade 1 (완전 마비): 정중선 완전 벗어남
- **난이도 특성**: 좌우 비대칭(bilateral asymmetry) → bilateral comparison feature가 효과적
- **데이터 부족**: 기존 라벨 192개로 가장 적음

### 안면_무표정

- 안정 시 좌우 대칭 평가 (resting symmetry)
- **난이도 특성**: 주관적 정의 → label smoothing이 특히 효과적
- **Class 분포**: Class 2가 70.4% 지배적 → focal loss 또는 class-balanced sampling 필요
- **데이터**: 기존 라벨 0개, 신규 라벨링으로만 확보

## Hard Action Resolution Guide

3개 Hard Action(이마_주름, 입_우, 안면_무표정)은 공통적으로:

1. **Class 2 지배 현상**: 60-75%가 중등도 → 모델이 Class 2로 수렴하는 경향
   - 해결: Focal loss, class-balanced sampling, ordinal-aware label smoothing
2. **작은 Test set**: 30-50 samples → MAE 분산 ±0.05, 큰 개선만 유의미
3. **주관적 평가 경계**: 라벨러 간 일치도가 낮을 수 있음 → soft label / label distribution learning

구체적 권장사항:
- 이마_주름: multi-scale features 또는 high-res crop → 미묘한 주름 차이 포착
- 안면_무표정: label smoothing → 주관적 평가 경계 모호성 완화
- 입_우: bilateral comparison features → 좌우 비대칭 직접 모델링
