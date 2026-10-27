"""Export functionality for masks, overlays, and metrics"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import cv2


def export_mask(mask: np.ndarray, output_path: str) -> None:
    """
    Export binary mask as PNG image
    
    Args:
        mask: Binary mask (0 or 1)
        output_path: Path to save the mask
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(mask_uint8).save(output_path)


def export_overlay(overlay: np.ndarray, output_path: str) -> None:
    """
    Export overlay image
    
    Args:
        overlay: Overlay image
        output_path: Path to save the overlay
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure RGB format
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(overlay).save(output_path)


def export_metrics_csv(
    metrics_dict: Dict[str, float],
    output_path: str,
    image_name: Optional[str] = None
) -> None:
    """
    Export metrics to CSV file
    
    Args:
        metrics_dict: Dictionary of metrics
        output_path: Path to save the CSV
        image_name: Optional image name for the record
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    if image_name:
        metrics_dict['image'] = image_name
    
    df = pd.DataFrame([metrics_dict])
    
    # Append if file exists, otherwise create new
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)


def export_metrics_batch(
    metrics_list: list,
    output_path: str
) -> None:
    """
    Export batch metrics to CSV file
    
    Args:
        metrics_list: List of metric dictionaries
        output_path: Path to save the CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)


def export_settings_json(
    settings: Dict,
    output_path: str
) -> None:
    """
    Export experiment settings as JSON
    
    Args:
        settings: Dictionary of settings
        output_path: Path to save the JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(settings, f, indent=2)


def export_results(
    predicted_mask: np.ndarray,
    overlay: np.ndarray,
    metrics: Dict[str, float],
    settings: Dict,
    output_dir: str,
    image_name: str = "result"
) -> Dict[str, str]:
    """
    Export all results (mask, overlay, metrics, settings)
    
    Args:
        predicted_mask: Predicted binary mask
        overlay: Overlay image
        metrics: Dictionary of metrics
        settings: Dictionary of settings
        output_dir: Output directory
        image_name: Base name for output files
        
    Returns:
        Dictionary of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_paths = {}
    
    # Export mask
    mask_path = output_dir / f"{image_name}_mask.png"
    export_mask(predicted_mask, mask_path)
    exported_paths['mask'] = str(mask_path)
    
    # Export overlay
    overlay_path = output_dir / f"{image_name}_overlay.png"
    export_overlay(overlay, overlay_path)
    exported_paths['overlay'] = str(overlay_path)
    
    # Export metrics
    metrics_path = output_dir / f"{image_name}_metrics.csv"
    export_metrics_csv(metrics, metrics_path, image_name)
    exported_paths['metrics'] = str(metrics_path)
    
    # Export settings
    settings_path = output_dir / f"{image_name}_settings.json"
    export_settings_json(settings, settings_path)
    exported_paths['settings'] = str(settings_path)
    
    return exported_paths
