"""Tests for preprocessing utilities"""

import numpy as np
import pytest
from src.preprocessing.image_utils import (
    normalize_image,
    rgb_to_grayscale,
    apply_filter,
    resize_image,
    ensure_binary_mask
)


def test_normalize_image_minmax():
    """Test min-max normalization"""
    img = np.array([[0, 127, 255]], dtype=np.uint8)
    normalized = normalize_image(img, method="minmax")
    
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized[0, 0] == pytest.approx(0.0)
    assert normalized[0, 2] == pytest.approx(1.0)


def test_normalize_image_zscore():
    """Test z-score normalization"""
    img = np.random.rand(10, 10).astype(np.float32)
    normalized = normalize_image(img, method="zscore")
    
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1e-6)


def test_rgb_to_grayscale():
    """Test RGB to grayscale conversion"""
    img_rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    img_gray = rgb_to_grayscale(img_rgb)
    
    assert img_gray.ndim == 2
    assert img_gray.shape == (10, 10)


def test_rgb_to_grayscale_already_gray():
    """Test that grayscale image is unchanged"""
    img_gray = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    result = rgb_to_grayscale(img_gray)
    
    assert np.array_equal(result, img_gray)


def test_apply_filter_gaussian():
    """Test Gaussian filter"""
    img = np.random.rand(50, 50).astype(np.float32)
    filtered = apply_filter(img, filter_type="gaussian", kernel_size=5)
    
    assert filtered.shape == img.shape
    assert not np.array_equal(filtered, img)  # Should be different


def test_apply_filter_median():
    """Test median filter"""
    img = np.random.rand(50, 50).astype(np.float32)
    filtered = apply_filter(img, filter_type="median", kernel_size=5)
    
    assert filtered.shape == img.shape


def test_resize_image_target_size():
    """Test resizing to target size"""
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    resized = resize_image(img, target_size=(50, 50))
    
    assert resized.shape == (50, 50, 3)


def test_resize_image_max_size():
    """Test resizing by max dimension"""
    img = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
    resized = resize_image(img, max_size=100)
    
    assert max(resized.shape[:2]) <= 100


def test_ensure_binary_mask():
    """Test binary mask conversion"""
    mask = np.array([[0, 100, 150, 200, 255]], dtype=np.uint8)
    binary = ensure_binary_mask(mask, threshold=127)
    
    assert binary.dtype == np.uint8
    assert set(binary.flatten()) == {0, 1}
    assert binary[0, 0] == 0
    assert binary[0, 4] == 1
