from __future__ import annotations


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, float]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(num_classes))
    accuracy = correct / total if total else 0.0

    f1_values: list[float] = []
    recall_values: list[float] = []
    support_values: list[int] = []
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes) if row != class_index)
        fn = sum(confusion[class_index][col] for col in range(num_classes) if col != class_index)
        support = tp + fn  # number of true examples of this class in y_true
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        support_values.append(support)

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / num_classes if num_classes else 0.0,
        "balanced_accuracy": sum(recall_values) / num_classes if num_classes else 0.0,
        # Per-class breakdown (index-aligned with the `classes` list passed elsewhere,
        # e.g. config["classes"] / CsvImageDataset.class_to_index). Kept as plain lists
        # here (not dict-by-name) so this module stays name-agnostic; callers that know
        # the class order (train.py) are responsible for zipping names <-> values.
        "per_class_f1": f1_values,
        "per_class_recall": recall_values,
        "per_class_support": support_values,
    }
