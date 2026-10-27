"""Image preprocessing utilities"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import cv2


def load_image(path: str, as_grayscale: bool = False) -> np.ndarray:
    """
    Load an image from file
    
    Args:
        path: Path to image file
        as_grayscale: Whether to convert to grayscale
        
    Returns:
        Image as numpy array
    """
    if as_grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized image
    """
    if method == "minmax":
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        else:
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
            return np.zeros_like(image, dtype=np.float32)
    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        return image - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Resize image to target size or scale by max dimension
    
    Args:
        image: Input image
        target_size: Target (width, height) or None
        max_size: Maximum dimension or None
        
    Returns:
        Resized image
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    elif max_size is not None:
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using standard weights
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        Grayscale image (H, W)
    """
    if image.ndim == 2:
        return image
    # Use OpenCV's conversion which uses standard weights
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def apply_filter(
    image: np.ndarray,
    filter_type: str = "gaussian",
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply smoothing filter to image
    
    Args:
        image: Input image
        filter_type: Type of filter ('gaussian' or 'median')
        kernel_size: Size of the kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def ensure_binary_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Ensure mask is binary (0 or 1)
    
    Args:
        mask: Input mask
        threshold: Threshold for binarization
        
    Returns:
        Binary mask with values 0 or 1
    """
    if mask.dtype == np.uint8 or mask.dtype == np.float32:
        binary = (mask > threshold).astype(np.uint8)
    else:
        binary = mask.astype(np.uint8)
    return binary
