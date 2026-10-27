"""Dataset loaders for LEVIR-CD and SYSU-CD datasets"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset


class LevirDataset(BaseDataset):
    """LEVIR-CD dataset loader"""
    
    def __init__(self, root_path: str = "./levir"):
        """
        Initialize LEVIR-CD dataset
        
        Args:
            root_path: Path to LEVIR-CD dataset root directory
        """
        super().__init__(root_path)
        self.dataset_name = "LEVIR-CD"
        self.description = "Building-related changes in satellite images"
    
    def _validate_dataset(self):
        """Validate LEVIR-CD dataset structure"""
        root = self.root_path
        
        # Check for test split
        test_a = root / "test" / "A"
        test_b = root / "test" / "B"
        test_label = root / "test" / "label"
        
        if not (test_a.exists() and test_b.exists() and test_label.exists()):
            raise ValueError(
                f"Invalid LEVIR-CD structure. Expected test/A, test/B, test/label directories in {root}"
            )
        
        # Load image pairs
        self.image_pairs = self._load_split(test_a, test_b, test_label)
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No image pairs found in {root}")
    
    def _load_split(self, a_dir: Path, b_dir: Path, label_dir: Path) -> List[Tuple[str, str, str]]:
        """Load image pairs from a split"""
        pairs = []
        a_files = sorted([f for f in a_dir.glob("*.png") if f.is_file()])
        
        for a_file in a_files:
            filename = a_file.name
            b_file = b_dir / filename
            label_file = label_dir / filename
            
            if b_file.exists() and label_file.exists():
                pairs.append((
                    str(a_file.relative_to(self.root_path)),
                    str(b_file.relative_to(self.root_path)),
                    str(label_file.relative_to(self.root_path))
                ))
        
        return pairs
    
    def get_image_pairs(self) -> List[Tuple[str, str, str]]:
        """Get all image pairs"""
        return self.image_pairs
    
    def get_split_names(self) -> List[str]:
        """Get available splits"""
        return ["test"]
    
    def get_image_pairs_by_split(self, split: str) -> List[Tuple[str, str, str]]:
        """Get image pairs for a specific split"""
        if split == "test":
            return self.image_pairs
        return []


class SysuDataset(BaseDataset):
    """SYSU-CD dataset loader"""
    
    def __init__(self, root_path: str = "./sysu"):
        """
        Initialize SYSU-CD dataset
        
        Args:
            root_path: Path to SYSU-CD dataset root directory
        """
        self.dataset_name = "SYSU-CD"
        self.description = "Change detection across urban and landscape scenes"
        self.splits = {}
        super().__init__(root_path)
    
    def _validate_dataset(self):
        """Validate SYSU-CD dataset structure"""
        root = self.root_path
        
        # Check for train, test, val splits
        for split_name in ["train", "test", "val"]:
            split_dir = root / split_name
            if split_dir.exists():
                time1_dir = split_dir / "time1"
                time2_dir = split_dir / "time2"
                label_dir = split_dir / "label"
                
                if time1_dir.exists() and time2_dir.exists() and label_dir.exists():
                    pairs = self._load_split(time1_dir, time2_dir, label_dir)
                    self.splits[split_name] = pairs
                    self.image_pairs.extend(pairs)
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No image pairs found in {root}")
    
    def _load_split(self, time1_dir: Path, time2_dir: Path, label_dir: Path) -> List[Tuple[str, str, str]]:
        """Load image pairs from a split"""
        pairs = []
        time1_files = sorted([f for f in time1_dir.glob("*.png") if f.is_file()])
        
        for time1_file in time1_files:
            filename = time1_file.name
            time2_file = time2_dir / filename
            label_file = label_dir / filename
            
            if time2_file.exists() and label_file.exists():
                pairs.append((
                    str(time1_file.relative_to(self.root_path)),
                    str(time2_file.relative_to(self.root_path)),
                    str(label_file.relative_to(self.root_path))
                ))
        
        return pairs
    
    def get_image_pairs(self) -> List[Tuple[str, str, str]]:
        """Get all image pairs"""
        return self.image_pairs
    
    def get_split_names(self) -> List[str]:
        """Get available splits"""
        return list(self.splits.keys())
    
    def get_image_pairs_by_split(self, split: str) -> List[Tuple[str, str, str]]:
        """Get image pairs for a specific split"""
        return self.splits.get(split, [])


def load_image_pair(
    earlier_path: str,
    later_path: str,
    label_path: Optional[str] = None,
    root_path: str = "."
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load an image pair and optionally its label
    
    Args:
        earlier_path: Relative path to earlier image
        later_path: Relative path to later image
        label_path: Optional relative path to label mask
        root_path: Root directory for the paths
        
    Returns:
        Tuple of (earlier_image, later_image, label_image or None)
    """
    root = Path(root_path)
    
    earlier_img = np.array(Image.open(root / earlier_path))
    later_img = np.array(Image.open(root / later_path))
    
    label_img = None
    if label_path:
        label_img = np.array(Image.open(root / label_path))
        # Convert to binary if needed
        if label_img.ndim == 3:
            label_img = label_img[:, :, 0]
        label_img = (label_img > 127).astype(np.uint8)
    
    return earlier_img, later_img, label_img
