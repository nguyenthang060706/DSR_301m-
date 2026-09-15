# Week 1 Protocol Lock

Status: data split completed on 2026-09-13. Final epoch budget, batch size, and LR schedule must be locked after the teacher/student convergence curves are available.

## Required Outputs

- Class distribution report for TrashNet 6 original classes.
- Fixed Dev/Corruption-holdout split, stratified by class.
- Fixed 5-fold CV indices over the remaining TrashNet pool, using identical fold indices for every internal ablation.
- TACO to TrashNet-6 mapping protocol.
- Teacher ResNet50 and student ResNet18 baseline metrics.
- Convergence-curve based decision on epoch budget, batch size, and LR schedule.

## Data Split Decision

Use a single fixed stratified split:

- 15% TrashNet -> Dev/Corruption-holdout.
- 85% TrashNet -> 5-fold CV pool.

The holdout split is never used in any fold training set. It is used for hyperparameter tuning and as the clean image source for TrashNet-C. Consistency weight `delta` must not be selected using severity-3 corruption scores.

Generated split files:

- `data/splits/trashnet_dev_corruption_holdout.csv`: 379 images.
- `data/splits/trashnet_cv_folds.csv`: 2148 images.

Observed TrashNet class counts:

| class | images |
| --- | ---: |
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| total | 2527 |

## Locked Internal Protocol

The final protocol is stored in `configs/week1_protocol.json` and is now **locked**:

- **Batch size**: 32
- **Optimizer**: SGD with momentum 0.9, weight decay 1e-4
- **Loss Function**: Weighted Cross-Entropy (Inverse-frequency class weights)
- **Max Epochs**: 80 (Safe budget covering convergence for both models)
- **LR schedule**: Cosine annealing (base 0.01, min 1e-6) with 5 warmup epochs.

### Baseline Results

Teacher (ResNet50) vs Student (ResNet18) clean validation metrics:

| Model | Peak Epoch | Val Accuracy | Val Macro-F1 | Val F1 (Trash) |
| --- | --- | --- | --- | --- |
| **Teacher (ResNet50)** | 43 | 94.45% | 94.48% | 95.00% |
| **Student (ResNet18)** | 73 | 92.34% | 91.37% | 85.00% |
| **Gap** | - | **2.11%** | **3.11%** | **10.0%** |

*Note: Since the accuracy gap is under 8 percentage points (~2.11%), KL divergence `KL(p_T || p_S)` analysis will be mandated during the Knowledge Distillation phase in Week 2 to explicitly measure knowledge transfer capacity.*

## TACO Mapping Protocol

Use the TrashNet six-class target space:

- cardboard;
- glass;
- metal;
- paper;
- plastic;
- trash.

Freeze mapping rules before any external evaluation:

- keep only TACO categories that can be mapped unambiguously to one of the six target classes;
- discard or mark as `ignore` mixed, composite, biological, hazardous, or ambiguous categories;
- never change the mapping after seeing model performance on TACO;
- store the final mapping table in `data/mappings/taco_to_trashnet6.csv`.

The current scaffold includes a draft mapping file that must be reviewed against the exact TACO category list before use.
