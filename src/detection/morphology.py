"""Morphological operations for change detection refinement"""

import numpy as np
import cv2


def morphological_opening(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation)
    
    Removes small objects and noise
    
    Args:
        binary_mask: Binary mask (0 or 1)
        kernel_size: Size of the structuring element
        iterations: Number of iterations
        
    Returns:
        Processed binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(
        (binary_mask * 255).astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )
    return (result > 0).astype(np.uint8)


def morphological_closing(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion)
    
    Fills small holes and gaps
    
    Args:
        binary_mask: Binary mask (0 or 1)
        kernel_size: Size of the structuring element
        iterations: Number of iterations
        
    Returns:
        Processed binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(
        (binary_mask * 255).astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )
    return (result > 0).astype(np.uint8)


def morphological_refinement(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    opening: bool = True,
    closing: bool = True,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological opening and closing for refinement
    
    Args:
        binary_mask: Binary mask (0 or 1)
        kernel_size: Size of the structuring element
        opening: Whether to apply opening
        closing: Whether to apply closing
        iterations: Number of iterations for each operation
        
    Returns:
        Refined binary mask
    """
    result = binary_mask.copy()
    
    if opening:
        result = morphological_opening(result, kernel_size, iterations)
    
    if closing:
        result = morphological_closing(result, kernel_size, iterations)
    
    return result
