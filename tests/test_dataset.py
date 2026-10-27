"""Tests for dataset loaders"""

import numpy as np
import pytest
from pathlib import Path
from src.data.dataset_loader import LevirDataset, SysuDataset, load_image_pair


def test_levir_dataset_initialization():
    """Test LEVIR-CD dataset initialization"""
    try:
        dataset = LevirDataset("./levir")
        assert dataset.dataset_name == "LEVIR-CD"
        assert len(dataset) > 0
    except ValueError:
        pytest.skip("LEVIR-CD dataset not available")


def test_levir_dataset_get_splits():
    """Test LEVIR-CD dataset split retrieval"""
    try:
        dataset = LevirDataset("./levir")
        splits = dataset.get_split_names()
        assert "test" in splits
    except ValueError:
        pytest.skip("LEVIR-CD dataset not available")


def test_levir_dataset_get_image_pairs():
    """Test LEVIR-CD dataset image pair retrieval"""
    try:
        dataset = LevirDataset("./levir")
        pairs = dataset.get_image_pairs()
        assert len(pairs) > 0
        assert len(pairs[0]) == 3  # (earlier, later, label)
    except ValueError:
        pytest.skip("LEVIR-CD dataset not available")


def test_sysu_dataset_initialization():
    """Test SYSU-CD dataset initialization"""
    try:
        dataset = SysuDataset("./sysu")
        assert dataset.dataset_name == "SYSU-CD"
        assert len(dataset) > 0
    except ValueError:
        pytest.skip("SYSU-CD dataset not available")


def test_sysu_dataset_get_splits():
    """Test SYSU-CD dataset split retrieval"""
    try:
        dataset = SysuDataset("./sysu")
        splits = dataset.get_split_names()
        assert len(splits) > 0
    except ValueError:
        pytest.skip("SYSU-CD dataset not available")


def test_sysu_dataset_get_image_pairs():
    """Test SYSU-CD dataset image pair retrieval"""
    try:
        dataset = SysuDataset("./sysu")
        pairs = dataset.get_image_pairs()
        assert len(pairs) > 0
        assert len(pairs[0]) == 3  # (time1, time2, label)
    except ValueError:
        pytest.skip("SYSU-CD dataset not available")


def test_load_image_pair():
    """Test loading an image pair"""
    try:
        dataset = LevirDataset("./levir")
        pairs = dataset.get_image_pairs()
        
        if len(pairs) > 0:
            earlier_path, later_path, label_path = pairs[0]
            earlier_img, later_img, label_img = load_image_pair(
                earlier_path, later_path, label_path, root_path=dataset.root_path
            )
            
            assert earlier_img is not None
            assert later_img is not None
            assert label_img is not None
            assert earlier_img.shape == later_img.shape
    except ValueError:
        pytest.skip("LEVIR-CD dataset not available")


def test_load_image_pair_without_label():
    """Test loading image pair without label"""
    try:
        dataset = LevirDataset("./levir")
        pairs = dataset.get_image_pairs()
        
        if len(pairs) > 0:
            earlier_path, later_path, _ = pairs[0]
            earlier_img, later_img, label_img = load_image_pair(
                earlier_path, later_path, None, root_path=dataset.root_path
            )
            
            assert earlier_img is not None
            assert later_img is not None
            assert label_img is None
    except ValueError:
        pytest.skip("LEVIR-CD dataset not available")
