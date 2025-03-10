# evaluator.py
import math
from typing import Set, Tuple, Dict

def evaluate(detected: Set[str], normal: Set[str], abnormal: Set[str]) -> Tuple[int, int, int, int]:
    """
    Evaluate detection results.
    Args:
        detected (Set[str]): Set of detected sequence IDs.
        normal (Set[str]): Set of normal sequence IDs.
        abnormal (Set[str]): Set of abnormal sequence IDs.
    Returns:
        Tuple[int, int, int, int]: TP, FN, TN, FP.
    """
    tp = len(detected.intersection(abnormal))
    fn = len(set(abnormal).difference(detected))
    tn = len(set(normal).difference(detected))
    fp = len(detected.intersection(normal))
    return tp, fn, tn, fp

def print_results(name: str, tp: int, fn: int, tn: int, fp: int, threshold: float, det_time: float) -> Dict[str, float]:
    """
    Print evaluation results.
    Args:
        name (str): Name of the detection method.
        tp (int): True positives.
        fn (int): False negatives.
        tn (int): True negatives.
        fp (int): False positives.
        threshold (float): Detection threshold.
        det_time (float): Detection time.
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 'inf'
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 'inf'
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 'inf'
    p = tp / (tp + fp) if (tp + fp) != 0 else 'inf'
    f1 = get_fone(tp, fn, tn, fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 'inf'
    print(f"{name}\n Threshold={threshold}\n Time={det_time}\n TP={tp}\n FP={fp}\n TN={tn}\n FN={fn}\n TPR=R={tpr}\n FPR={fpr}\n TNR={tnr}\n P={p}\n F1={f1}\n ACC={acc}\n MCC={mcc}")
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': f1, 'acc': acc, 'mcc': mcc}

def get_fone(tp: int, fn: int, tn: int, fp: int) -> float:
    """
    Compute F1 score.
    Args:
        tp (int): True positives.
        fn (int): False negatives.
        tn (int): True negatives.
        fp (int): False positives.
    Returns:
        float: F1 score.
    """
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))