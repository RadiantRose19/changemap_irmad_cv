"""Base dataset class for change detection datasets"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class BaseDataset(ABC):
    """Abstract base class for change detection datasets"""
    
    def __init__(self, root_path: str):
        """
        Initialize dataset
        
        Args:
            root_path: Path to the dataset root directory
        """
        self.root_path = Path(root_path)
        self.image_pairs = []
        self._validate_dataset()
    
    @abstractmethod
    def _validate_dataset(self):
        """Validate that the dataset structure is correct"""
        pass
    
    @abstractmethod
    def get_image_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get list of (earlier_image, later_image, label_image) tuples
        
        Returns:
            List of tuples containing relative paths to image pairs
        """
        pass
    
    @abstractmethod
    def get_split_names(self) -> List[str]:
        """Get available split names (e.g., ['train', 'test', 'val'])"""
        pass
    
    def get_image_pairs_by_split(self, split: str) -> List[Tuple[str, str, str]]:
        """
        Get image pairs for a specific split
        
        Args:
            split: Split name (e.g., 'train', 'test', 'val')
            
        Returns:
            List of tuples containing relative paths to image pairs
        """
        all_pairs = self.get_image_pairs()
        # Filter by split if needed (implementation depends on dataset structure)
        return all_pairs
    
    def __len__(self) -> int:
        """Return number of image pairs"""
        return len(self.image_pairs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root_path}, pairs={len(self)})"
