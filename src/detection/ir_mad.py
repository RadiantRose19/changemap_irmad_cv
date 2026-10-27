"""Iteratively Reweighted Multivariate Alteration Detection (IR-MAD)"""

import numpy as np
from scipy.linalg import eigh
from ..preprocessing.image_utils import rgb_to_grayscale, normalize_image


def ir_mad(
    img1: np.ndarray,
    img2: np.ndarray,
    max_iterations: int = 30,
    convergence_threshold: float = 1e-6
) -> tuple:
    """
    Iteratively Reweighted Multivariate Alteration Detection (IR-MAD)
    
    This algorithm performs Canonical Correlation Analysis (CCA) between two images
    and iteratively reweights observations to reduce the influence of changed pixels.
    
    Args:
        img1: First image (earlier), shape (H, W) or (H, W, C)
        img2: Second image (later), shape (H, W) or (H, W, C)
        max_iterations: Maximum number of iterations
        convergence_threshold: Convergence threshold for weights
        
    Returns:
        Tuple of (change_probability_image, mad_variates, weights)
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
    
    # Reshape to (n_pixels, 1) for CCA
    h, w = img1_norm.shape
    n_pixels = h * w
    
    X = img1_norm.reshape(n_pixels, 1)
    Y = img2_norm.reshape(n_pixels, 1)
    
    # Initialize weights (all pixels have equal weight initially)
    weights = np.ones(n_pixels)
    
    # Store previous weights for convergence check
    prev_weights = weights.copy()
    
    for iteration in range(max_iterations):
        # Compute weighted covariance matrices
        W = np.diag(weights)
        total_weight = weights.sum()
        
        # Center the data
        X_mean = (weights @ X) / total_weight
        Y_mean = (weights @ Y) / total_weight
        
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        
        # Compute covariance matrices
        Cxx = (X_centered.T @ W @ X_centered) / total_weight
        Cyy = (Y_centered.T @ W @ Y_centered) / total_weight
        Cxy = (X_centered.T @ W @ Y_centered) / total_weight
        
        # Add small regularization to avoid singular matrices
        reg = 1e-10
        Cxx += reg * np.eye(Cxx.shape[0])
        Cyy += reg * np.eye(Cyy.shape[0])
        
        # Solve generalized eigenvalue problem for CCA
        # Cxx^-1 * Cxy * Cyy^-1 * Cyx * a = rho^2 * a
        try:
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            Cyx = Cxy.T
            
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
            eigenvalues, eigenvectors = eigh(M)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Compute MAD variates
            a = eigenvectors[:, 0:1]
            b = Cyy_inv @ Cyx @ a / np.sqrt(eigenvalues[0])
            
            # Normalize vectors
            a = a / np.sqrt(a.T @ Cxx @ a)
            b = b / np.sqrt(b.T @ Cyy @ b)
            
            # Compute MAD variates
            mad1 = (X_centered @ a).flatten()
            mad2 = (Y_centered @ b).flatten()
            
            # Compute change magnitude from MAD variates
            mad_magnitude = np.sqrt(mad1**2 + mad2**2)
            
            # Update weights based on MAD magnitude (inverse weighting)
            # Pixels with large MAD magnitude are likely changed and get lower weight
            mad_std = mad_magnitude.std()
            if mad_std > 0:
                weights = np.exp(-(mad_magnitude**2) / (2 * mad_std**2))
            else:
                weights = np.ones_like(weights)
            
            # Check convergence
            weight_diff = np.abs(weights - prev_weights).max()
            if weight_diff < convergence_threshold:
                break
                
            prev_weights = weights.copy()
            
        except np.linalg.LinAlgError:
            # If matrix inversion fails, use simple difference
            print(f"Warning: Matrix inversion failed at iteration {iteration}, using fallback")
            mad_magnitude = np.abs(X.flatten() - Y.flatten())
            weights = np.ones_like(weights)
            break
    
    # Compute final change probability image
    # Use normalized MAD magnitude as change probability
    if mad_magnitude.max() > 0:
        change_probability = mad_magnitude / mad_magnitude.max()
    else:
        change_probability = np.zeros_like(mad_magnitude)
    
    # Reshape back to image dimensions
    change_probability = change_probability.reshape(h, w)
    
    return change_probability, mad_magnitude.reshape(h, w), weights.reshape(h, w)
