"""Thresholding algorithms for change detection"""

import numpy as np
import cv2


def otsu_threshold(image: np.ndarray) -> tuple:
    """
    Apply Otsu's thresholding method
    
    Args:
        image: Input grayscale image
        
    Returns:
        Tuple of (threshold, binary_mask)
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Apply Otsu's thresholding
    threshold, binary = cv2.threshold(
        image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Normalize threshold back to [0, 1] if input was normalized
    if image.max() <= 1.0:
        threshold = threshold / 255.0
    
    return threshold, (binary > 0).astype(np.uint8)


def fixed_threshold(
    image: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply fixed threshold
    
    Args:
        image: Input grayscale image
        threshold: Threshold value (0-1 for normalized, 0-255 for uint8)
        
    Returns:
        Binary mask
    """
    # Determine if image is normalized or uint8
    if image.max() <= 1.0:
        # Normalized image
        binary = (image > threshold).astype(np.uint8)
    else:
        # uint8 image
        if threshold <= 1.0:
            threshold = threshold * 255
        binary = (image > threshold).astype(np.uint8)
    
    return binary
