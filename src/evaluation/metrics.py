"""Evaluation metrics for change detection"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1: float
    iou: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
        }


def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray
) -> MetricsResult:
    """
    Calculate evaluation metrics for change detection
    
    Args:
        prediction: Predicted binary mask (0 or 1)
        ground_truth: Ground truth binary mask (0 or 1)
        
    Returns:
        MetricsResult with precision, recall, f1, and iou
    """
    # Ensure binary masks
    pred = (prediction > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    
    # Calculate confusion matrix elements
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # Precision: TP / (TP + FP)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    # Recall: TP / (TP + FN)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    # F1-score: 2 * (Precision * Recall) / (Precision + Recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # IoU (Intersection over Union): TP / (TP + FP + FN)
    if tp + fp + fn > 0:
        iou = tp / (tp + fp + fn)
    else:
        iou = 0.0
    
    return MetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou
    )


def calculate_metrics_batch(
    predictions: list,
    ground_truths: list
) -> Dict[str, float]:
    """
    Calculate average metrics across multiple predictions
    
    Args:
        predictions: List of predicted binary masks
        ground_truths: List of ground truth binary masks
        
    Returns:
        Dictionary with average metrics
    """
    assert len(predictions) == len(ground_truths), "Number of predictions and ground truths must match"
    
    metrics_list = []
    for pred, gt in zip(predictions, ground_truths):
        metrics = calculate_metrics(pred, gt)
        metrics_list.append(metrics)
    
    # Calculate averages
    avg_precision = np.mean([m.precision for m in metrics_list])
    avg_recall = np.mean([m.recall for m in metrics_list])
    avg_f1 = np.mean([m.f1 for m in metrics_list])
    avg_iou = np.mean([m.iou for m in metrics_list])
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "iou": avg_iou,
    }
