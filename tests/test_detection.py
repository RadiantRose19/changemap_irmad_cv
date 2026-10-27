"""Tests for change detection algorithms"""

import numpy as np
import pytest
from src.detection.differencing import absolute_difference
from src.detection.thresholding import otsu_threshold, fixed_threshold
from src.detection.morphology import morphological_opening, morphological_closing, morphological_refinement
from src.detection.cva import change_vector_analysis
from src.detection.ir_mad import ir_mad
from src.detection.pipelines import BaselinePipeline, CVAPipeline, IRMADPipeline, get_pipeline


def test_absolute_difference():
    """Test absolute difference calculation"""
    img1 = np.ones((10, 10), dtype=np.uint8) * 100
    img2 = np.ones((10, 10), dtype=np.uint8) * 150
    
    diff = absolute_difference(img1, img2, use_filter=False)
    
    assert diff.shape == (10, 10)
    assert diff.max() == pytest.approx(1.0)  # Normalized


def test_absolute_difference_rgb():
    """Test absolute difference with RGB images"""
    img1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 150
    
    diff = absolute_difference(img1, img2, use_filter=False)
    
    assert diff.shape == (10, 10)


def test_otsu_threshold():
    """Test Otsu thresholding"""
    img = np.concatenate([
        np.zeros((5, 10), dtype=np.uint8),
        np.ones((5, 10), dtype=np.uint8) * 255
    ], axis=0)
    
    threshold, binary = otsu_threshold(img)
    
    assert 0 <= threshold <= 1.0 or 0 <= threshold <= 255
    assert binary.dtype == np.uint8
    assert set(binary.flatten()) == {0, 1}


def test_fixed_threshold():
    """Test fixed thresholding"""
    img = np.random.rand(10, 10).astype(np.float32)
    
    binary = fixed_threshold(img, threshold=0.5)
    
    assert binary.dtype == np.uint8
    assert set(binary.flatten()) == {0, 1}


def test_morphological_opening():
    """Test morphological opening"""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    mask[4, 4] = 0  # Small hole
    
    opened = morphological_opening(mask, kernel_size=3)
    
    assert opened.shape == mask.shape
    assert opened.dtype == np.uint8


def test_morphological_closing():
    """Test morphological closing"""
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 0
    mask[4, 4] = 1  # Small object in hole
    
    closed = morphological_closing(mask, kernel_size=3)
    
    assert closed.shape == mask.shape
    assert closed.dtype == np.uint8


def test_morphological_refinement():
    """Test full morphological refinement"""
    mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
    
    refined = morphological_refinement(mask, kernel_size=3, opening=True, closing=True)
    
    assert refined.shape == mask.shape
    assert refined.dtype == np.uint8


def test_change_vector_analysis():
    """Test Change Vector Analysis"""
    img1 = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    
    change_magnitude = change_vector_analysis(img1, img2)
    
    assert change_magnitude.ndim == 2
    assert change_magnitude.shape == (10, 10)
    assert change_magnitude.min() >= 0.0


def test_ir_mad():
    """Test IR-MAD algorithm"""
    img1 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    change_prob, mad_magnitude, weights = ir_mad(img1, img2, max_iterations=5)
    
    assert change_prob.shape == (50, 50)
    assert mad_magnitude.shape == (50, 50)
    assert weights.shape == (50, 50)
    assert change_prob.min() >= 0.0
    assert change_prob.max() <= 1.0


def test_baseline_pipeline():
    """Test baseline pipeline"""
    pipeline = BaselinePipeline()
    
    img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    binary_mask, intermediate = pipeline.run(img1, img2)
    
    assert binary_mask.shape == (50, 50)
    assert binary_mask.dtype == np.uint8
    assert 'difference' in intermediate
    assert 'refined' in intermediate


def test_cva_pipeline():
    """Test CVA pipeline"""
    pipeline = CVAPipeline()
    
    img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    binary_mask, intermediate = pipeline.run(img1, img2)
    
    assert binary_mask.shape == (50, 50)
    assert binary_mask.dtype == np.uint8
    assert 'change_magnitude' in intermediate


def test_ir_mad_pipeline():
    """Test IR-MAD pipeline"""
    pipeline = IRMADPipeline()
    
    img1 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    
    binary_mask, intermediate = pipeline.run(img1, img2, max_iterations=5)
    
    assert binary_mask.shape == (50, 50)
    assert binary_mask.dtype == np.uint8
    assert 'change_probability' in intermediate


def test_get_pipeline():
    """Test pipeline factory"""
    baseline = get_pipeline("baseline")
    cva = get_pipeline("cva")
    ir_mad = get_pipeline("ir_mad")
    
    assert isinstance(baseline, BaselinePipeline)
    assert isinstance(cva, CVAPipeline)
    assert isinstance(ir_mad, IRMADPipeline)


def test_get_pipeline_invalid():
    """Test pipeline factory with invalid name"""
    with pytest.raises(ValueError):
        get_pipeline("invalid_pipeline")
