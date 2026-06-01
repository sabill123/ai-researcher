# Loss Function 가이드

## 현재 사용 중

### MSE Loss
- 기본 regression loss
- Severity 연속값 예측에 적합
- 단점: class 불균형에 취약, outlier에 민감

### Weighted Kappa Loss (WK)
- Quadratic weighted kappa를 loss로 변환
- Ordinal 관계 반영 (1등급 차이 < 2등급 차이)
- MSE와 결합 시 효과적

### CLOC Loss (Multi-Margin N-pair Loss)
- 인접 클래스 경계에 다른 margin 부여
- high margin [2.0, 1.0, 0.8, 0.6]: 정상/비정상 경계 강조
- Two-phase training: (1) encoder+margins 학습, (2) margins 고정
- 눈 동작 개선에 효과적, 입 동작은 주의 필요

### Cross Entropy (CE)
- Action classification auxiliary loss
- do_action_classification=True일 때 활성화

## 유망한 Loss Functions

### Focal Loss
- Hard example에 더 높은 weight → rare class(0,4) 학습 강제
- Class 2 지배 문제의 직접적 해결책
- α (class weight) + γ (focusing parameter) 튜닝 필요
- **적용 위치**: losses.py에 추가, train_cloc_v2.py의 training loop에서 호출

### Ordinal-aware Label Smoothing
- Hard label [0,0,1,0,0] → Soft label [0.02, 0.08, 0.75, 0.08, 0.02]
- 인접 class에 더 높은 확률 부여 (ordinal 구조 반영)
- 주관적 평가 경계가 모호한 action(안면_무표정)에 특히 효과적

### Label Distribution Learning (LDL)
- 각 sample에 대해 severity distribution 전체를 학습
- KL divergence 또는 JS divergence 기반 loss
- Gaussian label distribution: μ=실제severity, σ=불확실성

### Balanced MSE
- Class frequency 기반 weight 부여
- weight[c] = 1 / (class_count[c] / total_count)
- 간단하지만 Class 2 지배 완화에 효과적

### Consistency Regularization
- Augmented view 간 예측 일관성 강제
- 소규모 데이터셋의 generalization 개선

## Loss 조합 가이드

현재 가장 효과적인 조합:
```
Total Loss = MSE + λ₁·WK + λ₂·CLOC + [optional: λ₃·Focal/LDL]
```

- MSE: 기본 regression 신호
- WK: Ordinal 구조 유지
- CLOC: Class boundary 분리 강화
- Focal/LDL: Class 불균형 완화 (추가 시 실험 필요)

**주의**: Loss 항목 3개 이상 결합 시 gradient conflict 가능 → weight 신중히 조절
