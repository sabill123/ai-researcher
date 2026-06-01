# 데이터 제약 사항 및 분석 가이드

## 데이터셋 개요

- **총 학습 데이터**: ~2,435명 (기존 1,802명 + 신규 633명)
- **6개 TARGET ACTION**: 눈_살짝감기, 눈_질끈감기, 이마_주름, 입_이, 입_우, 안면_무표정
- **Severity**: 0-4 (Modified Sunnybrook Scale)
- **Test set**: Bradley-Terry 200장 (4개 action × 50장)

## 핵심 제약

### 1. 절대적 데이터 부족
- ~2,400명, 동작당 400~800장
- 복잡한 방법론(Triplet, 대규모 contrastive learning) 적용 어려움
- **권장**: 파라미터 효율적 접근 (LoRA, small head), 강한 regularization

### 2. 클래스 불균형 (Class 2 지배 현상)
- 전체: Severity 2 ~44%, Severity 0 ~2%, Severity 4 ~2%
- 이마_주름: Class 2 = 65.8%
- 안면_무표정: Class 2 = 70.4%
- 눈_질끈감기: Class 2 = 75.6%
- **결과**: 모델이 Class 2만 예측하는 trivial solution 수렴 경향
- **권장**: Focal loss, class-balanced sampling, weighted loss

### 3. 작은 Test set (통계적 한계)
- 이마_주름: ~30 test samples → MAE 분산 ±0.05
- 안면_무표정: ~30 test samples
- 입_우: ~40 test samples
- **의미**: 0.02 이하의 MAE 차이는 통계적으로 무의미할 수 있음
- **권장**: 큰 개선(>0.03)만 유의미한 진전으로 간주

### 4. 라벨 품질 이슈
- 4명 라벨러 간 일치도 차이
- 특히 안면_무표정, 이마_주름: 주관적 평가 경계
- **권장**: Label smoothing, label distribution learning으로 불확실성 반영

### 5. 도메인 차이
- 기존 데이터 (1,802명): 원래 병원 촬영 조건
- 신규 데이터 (633명, 강동/회기): 다른 촬영 환경
- **권장**: Domain-balanced sampling (batch 내 기존/신규 50:50)

## 분석 가이드

실험 결과를 분석할 때 다음을 고려:

1. **Action별 분해**: 평균 MAE 개선이 특정 action에만 집중됐는지 확인
2. **Class 분포 대비 성능**: Class 2 dominant action에서 MAE가 낮다면, Class 2만 예측하는 것일 수 있음
3. **통계적 유의성**: Test set 크기가 30-50인 action의 MAE 변화 0.02 이하는 noise일 가능성
4. **Technique type별 효과**: 같은 technique type의 연속 실패는 근본적 한계 시사
5. **Parent 대비 delta**: Incremental 변경의 효과가 +0.01 이상이면 해당 방향 재고 필요
6. **Overfitting 징후**: Val loss 상승 + Train loss 하강 패턴 주시
