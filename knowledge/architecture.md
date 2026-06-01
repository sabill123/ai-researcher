# 모델 아키텍처 및 코드 구조

## 모델 요약

- **Backbone**: FaRL ViT-B/16 (20M face image-text pairs pretrained, ICLR 2022)
- **Fine-tuning**: LoRA (rank=16, blocks 9-11) — 전체 파라미터의 ~2% 학습
- **Feature Fusion**: 중간 레이어 [3,5,7,11] aggregate + LayerNorm
- **Head**: SeparateHeadWithFeatures — 6개 action별 독립 MLP head
- **Loss**: MSE + Weighted Kappa + CLOC (Multi-Margin N-pair Loss)
- **입력**: 512x512 (YOLOv8 face crop 후 resize)

## 코드 구조

- `train_cloc_v2.py`: 학습 루프 (argparse CLI, train/val/test split, epoch loop)
- `model.py`: SeparateHeadWithFeatures (FaRL backbone + per-action MLP heads)
- `losses.py`: MSELoss, WeightedKappaLoss, CrossEntropyLoss, SeverityLossCombiner
- `cloc_loss.py`: CLOCLoss (contrastive ordinal), OrdinalContrastiveLoss
- `dataset.py`: KHUPalsyDataset (image loading, augmentation pipeline)
- `backbone.py`: FaRLEncoder (ViT-B/16, LoRA adapter, pretrained checkpoint loading)
- `utils.py`: READ-ONLY (metrics computation — score_mae, spearman_r 등)

## 제약사항

1. **CLI 인터페이스 유지 필수**: `--data_dir`, `--output_dir`, `--exp`, `--epochs`, `--no_wandb`
2. **test_metrics.json 출력 필수**: `*_score_mae` 키 포함
3. **utils.py는 READ-ONLY**: 메트릭 계산 로직 불변
4. **argparse choices 변경 금지**:
   - `severity_loss_fn`: ONLY "CE", "MSE", "MSE+WK"
   - `model_type`: ONLY "separate", "shared"
   - `cloc_type`: ONLY "cloc", "ordinal"
   - `margin_mode`: ONLY "single", "multi"
5. 새 loss/architecture는 코드 내부에서 직접 구현, argparse choices 추가 불가
6. 최대 500 lines 코드 변경

## 변경 가능 영역

- **Architecture**: 새 head 디자인, attention mechanism, feature aggregation 방식
- **Loss functions**: 새 loss, 조합 변경, class-balanced loss 추가
- **Augmentation**: 새 transforms, mixup, cutmix, face-specific augmentation
- **Training**: LR schedule, optimizer 변경, curriculum learning
- **Backbone**: Layer freezing 전략, adapter modules, LoRA rank 조정
