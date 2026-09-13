# DSR Research Implementation

Implementation workspace for the research plan in `ke_hoach_nghien_cuu_MASTER_v8.md`.

The week-1 goal is to lock the experimental protocol before training:

- prepare TrashNet in the original 6 classes;
- inspect class imbalance;
- create a fixed stratified Dev/Corruption-holdout split;
- reserve the remaining data for identical 5-fold CV indices;
- freeze the TACO-to-6-class mapping protocol;
- train the ResNet50 teacher and ResNet18 student baseline once dependencies and data are available;
- use convergence curves from those runs to finalize the epoch budget, batch size, and LR schedule.

## Expected Data Layout

Place TrashNet images under:

```text
data/raw/trashnet/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

The generated split files are written under `data/splits/`.

## Week-1 Commands

Inspect the dataset:

```powershell
python scripts/inspect_dataset.py --data-root data/raw/trashnet --out reports/week1/class_distribution.csv
```

Create the fixed holdout and CV folds:

```powershell
python scripts/make_trashnet_splits.py --data-root data/raw/trashnet --out-dir data/splits --holdout-ratio 0.15 --folds 5 --seed 20260913
```

Train baselines after installing the ML dependencies:

```powershell
python src/dsr/train.py --config configs/week1_protocol.json --model resnet50 --run-name teacher_resnet50_week1
python src/dsr/train.py --config configs/week1_protocol.json --model resnet18 --run-name student_resnet18_week1
```

## Current Status

This workspace currently contains the research plan and implementation scaffold. The actual week-1 training is pending local availability of TrashNet and PyTorch/torchvision.
