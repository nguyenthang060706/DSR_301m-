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
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes) if row != class_index)
        fn = sum(confusion[class_index][col] for col in range(num_classes) if col != class_index)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        recall_values.append(recall)

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / num_classes if num_classes else 0.0,
        "balanced_accuracy": sum(recall_values) / num_classes if num_classes else 0.0,
    }
