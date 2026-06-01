# 실험 기법 정리

## 효과 있었던 방법

### CLOC (Multi-Margin N-pair Loss) — CVPR 2025
- 인접 클래스 간 경계의 중요도 차등 반영
- high margin [2.0, 1.0, 0.8, 0.6]: 눈 동작 MAE 크게 개선 (22.6%↓)
- learnable margins + two-phase training이 효과적
- **주의**: 입 동작에서는 오히려 악화 가능 → action별 margin 튜닝 필요

### Separate Head Architecture
- Action별 독립 MLP head가 shared head보다 일관되게 우수
- 각 action의 고유한 특성을 개별 학습

### LoRA Fine-tuning (rank=16, blocks 9-11)
- Full fine-tuning 대비 overfitting 감소
- 소규모 데이터셋(~2400명)에 적합한 파라미터 효율적 접근

### Feature Fusion [3,5,7,11]
- 중간 레이어 aggregate가 final layer만 사용하는 것보다 우수
- 다양한 추상화 수준의 feature 결합

## 효과 없었던 방법

### Ranking Loss
- Spearman's r 향상, BUT Score MAE 악화 → trade-off

### Triplet Loss
- 심한 Overfitting (소규모 데이터셋에서 sampling 부족)

### Contrastive Learning (vanilla)
- 효과 미미, CLOC의 ordinal-aware 버전이 훨씬 효과적

### SORD, CORAL, OLL (Ordinal Loss 계열)
- 효과 미미 — 데이터 부족이 근본 원인일 가능성

### Mixup (vanilla)
- 효과 미미 — face-specific augmentation이 더 적합

### SAM Optimizer
- 효과 미미 — 추가 연산 대비 이득 없음

## 🔄 연구 방향 전환: 쌍대비교 기반 이미지 랭킹 (2026-03 미팅)

### 배경
- 절대적 중증도 채점(0-4)은 평가자 간 일치도가 낮아 MAE 0.49 달성이 구조적으로 어려움
- 쌍대비교("두 이미지 중 어느 쪽이 더 심한가?")는 일치도가 높음
- Pairwise → Ranking → Calibrated Score 파이프라인으로 전환

### 핵심 파이프라인
1. **Pairwise Comparison Model**: Siamese/Cross-attention 네트워크로 이미지 쌍 비교
2. **Ranking Aggregation**: Bradley-Terry / TrueSkill / Elo로 전체 순위 생성
3. **Score Calibration**: 순위 → 연속 중증도 점수로 변환 (isotonic regression 등)

### 탐색 키워드
- Bradley-Terry model, TrueSkill ranking, Elo rating for images
- Pairwise learning to rank, Siamese network for medical image comparison
- Image quality assessment (IQA) ranking — 유사한 문제 구조
- Preference learning, RLHF reward model (pairwise comparison 구조 동일)

### 연구 범위 확장
- "안면마비 중증도" → "이미지 랭킹" 일반 파이프라인으로 확장
- 다른 의료 영상 grading 문제에도 적용 가능한 범용 프레임워크 지향

## 유망한 미시도 기법

### Focal Loss + Class-balanced Sampling
- Severity 0/4 (rare class) 학습 강제
- Class 2 지배 현상 해결의 핵심

### Label Distribution Learning (LDL)
- Hard label 대신 soft label distribution 학습
- 주관적 평가 경계(안면_무표정 등)에 특히 효과적

### DIOR-ViT (Differential Ordinal Learning)
- 샘플 쌍 간 중증도 차이 학습 → robust ordinal 관계 학습
- ViT 기반 + medical image 검증 완료

### Deep Imbalanced Regression (LDS/FDS)
- Label Distribution Smoothing + Feature Distribution Smoothing
- 연속적 중증도 예측의 불균형 문제 해결

### Bilateral Comparison Features
- 좌우 얼굴 비대칭 직접 모델링
- 입_우, 안면_무표정 등 대칭성 평가 액션에 특히 유효

## 참고 논문

| 논문 | 학회/연도 | 핵심 아이디어 |
|------|-----------|---------------|
| CLOC | CVPR 2025 | Multi-Margin N-pair Loss |
| Hybrid Contrastive Ordinal | MICCAI 2025 | Distance-based prototype contrastive ordinal loss |
| DIOR-ViT | MedIA 2025 | Differential ordinal learning for ViT |
| ConOrd | 2025 | Soft affinity/disparity contrastive order loss |
| LS+ | MICCAI 2024 | Informed label smoothing for medical images |
| DIR | NeurIPS 2021 | LDS/FDS for imbalanced regression |
