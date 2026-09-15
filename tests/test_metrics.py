import math
from dsr.metrics import classification_metrics
from dsr.data import compute_class_weights

def test_macro_f1_hides_minority_failure():
    """
    Test simulating the exact scenario feared:
    High macro-f1 overall, but complete failure on minority class.
    """
    # 3 classes: A, B, C (minority)
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    # Predicts A and B perfectly, but gets all C wrong (guesses A instead)
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    
    metrics = classification_metrics(y_true, y_pred, num_classes=3)
    
    # Class 0 (A): TP=4, FP=2, FN=0 -> Prec=4/6, Rec=4/4 -> F1 = 0.8
    # Class 1 (B): TP=4, FP=0, FN=0 -> Prec=4/4, Rec=4/4 -> F1 = 1.0
    # Class 2 (C): TP=0, FP=0, FN=2 -> Prec=0, Rec=0 -> F1 = 0.0
    # Expected Macro F1: (0.8 + 1.0 + 0.0) / 3 = 0.6
    
    assert math.isclose(metrics["macro_f1"], 0.6)
    
    # The per-class breakdown catches what macro misses
    assert metrics["per_class_f1"][2] == 0.0
    assert metrics["per_class_support"][2] == 2
