"""Change Vector Analysis (CVA) for change detection"""

import numpy as np
from ..preprocessing.image_utils import rgb_to_grayscale, normalize_image


def change_vector_analysis(
    img1: np.ndarray,
    img2: np.ndarray
) -> np.ndarray:
    """
    Compute Change Vector Analysis magnitude
    
    Args:
        img1: First image (earlier)
        img2: Second image (later)
        
    Returns:
        Change magnitude image
    """
    # Ensure both images have same shape
    assert img1.shape == img2.shape, "Images must have same dimensions"
    
    # Convert to grayscale if needed
    if img1.ndim == 3:
        img1_gray = rgb_to_grayscale(img1).astype(np.float32)
        img2_gray = rgb_to_grayscale(img2).astype(np.float32)
    else:
        img1_gray = img1.astype(np.float32)
        img2_gray = img2.astype(np.float32)
    
    # Normalize to [0, 1]
    img1_norm = normalize_image(img1_gray)
    img2_norm = normalize_image(img2_gray)
    
    # Compute change magnitude (Euclidean distance)
    change_magnitude = np.sqrt(np.sum((img2_norm - img1_norm) ** 2, axis=-1) if img1_norm.ndim == 3 else (img2_norm - img1_norm) ** 2)
    
    return change_magnitude
