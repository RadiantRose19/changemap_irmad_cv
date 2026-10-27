"""Image differencing for change detection"""

import numpy as np
from ..preprocessing.image_utils import rgb_to_grayscale, apply_filter, normalize_image


def absolute_difference(
    img1: np.ndarray,
    img2: np.ndarray,
    use_filter: bool = True,
    filter_type: str = "gaussian",
    filter_size: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute absolute difference between two images
    
    Args:
        img1: First image (earlier)
        img2: Second image (later)
        use_filter: Whether to apply smoothing filter
        filter_type: Type of filter ('gaussian' or 'median')
        filter_size: Size of filter kernel
        normalize: Whether to normalize result to [0, 1]
        
    Returns:
        Difference image
    """
    # Convert to grayscale if needed
    if img1.ndim == 3:
        img1_gray = rgb_to_grayscale(img1)
    else:
        img1_gray = img1.copy()
    
    if img2.ndim == 3:
        img2_gray = rgb_to_grayscale(img2)
    else:
        img2_gray = img2.copy()
    
    # Apply optional filtering
    if use_filter:
        img1_gray = apply_filter(img1_gray, filter_type, filter_size)
        img2_gray = apply_filter(img2_gray, filter_type, filter_size)
    
    # Compute absolute difference
    diff = np.abs(img1_gray.astype(np.float32) - img2_gray.astype(np.float32))
    
    # Normalize to [0, 1]
    if normalize:
        if diff.max() > 0:
            diff = diff / diff.max()
    
    return diff
