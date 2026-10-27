"""Tests for evaluation metrics"""

import numpy as np
import pytest
from src.evaluation.metrics import calculate_metrics, calculate_metrics_batch, MetricsResult


def test_calculate_metrics_perfect():
    """Test metrics calculation with perfect prediction (all changes detected)"""
    prediction = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    ground_truth = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)


def test_calculate_metrics_no_change():
    """Test metrics with no change in either"""
    prediction = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    ground_truth = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    # No TP, FP, or FN - should handle gracefully
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0


def test_calculate_metrics_partial_match():
    """Test metrics with partial match"""
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    ground_truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    # TP=1, FP=1, FN=1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.iou == pytest.approx(1/3, rel=1e-6)


def test_calculate_metrics_all_false_positives():
    """Test metrics with all false positives"""
    prediction = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    ground_truth = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0


def test_calculate_metrics_all_false_negatives():
    """Test metrics with all false negatives"""
    prediction = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    ground_truth = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0


def test_calculate_metrics_non_binary_input():
    """Test metrics with non-binary input (should be converted)"""
    prediction = np.array([[0.2, 0.8], [0.3, 0.9]], dtype=np.float32)
    ground_truth = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    
    metrics = calculate_metrics(prediction, ground_truth)
    
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.iou <= 1.0


def test_metrics_result_to_dict():
    """Test MetricsResult to_dict conversion"""
    metrics = MetricsResult(precision=0.8, recall=0.7, f1=0.75, iou=0.6)
    metrics_dict = metrics.to_dict()
    
    assert isinstance(metrics_dict, dict)
    assert metrics_dict['precision'] == 0.8
    assert metrics_dict['recall'] == 0.7
    assert metrics_dict['f1'] == 0.75
    assert metrics_dict['iou'] == 0.6


def test_calculate_metrics_batch():
    """Test batch metrics calculation"""
    predictions = [
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    ]
    ground_truths = [
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
    ]
    
    avg_metrics = calculate_metrics_batch(predictions, ground_truths)
    
    assert isinstance(avg_metrics, dict)
    assert 'precision' in avg_metrics
    assert 'recall' in avg_metrics
    assert 'f1' in avg_metrics
    assert 'iou' in avg_metrics
    assert all(0.0 <= v <= 1.0 for v in avg_metrics.values())


def test_calculate_metrics_batch_mismatch():
    """Test batch metrics with mismatched lengths"""
    predictions = [np.array([[1, 0]], dtype=np.uint8)]
    ground_truths = [
        np.array([[1, 0]], dtype=np.uint8),
        np.array([[0, 1]], dtype=np.uint8),
    ]
    
    with pytest.raises(AssertionError):
        calculate_metrics_batch(predictions, ground_truths)
