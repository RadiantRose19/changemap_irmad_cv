"""Visualization utilities for change detection"""

import numpy as np
import cv2
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def create_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay of binary mask on image
    
    Args:
        image: Base image (H, W) or (H, W, 3)
        binary_mask: Binary mask (H, W) with values 0 or 1
        color: RGB color for overlay (default: red)
        alpha: Transparency of overlay (0-1)
        
    Returns:
        Overlay image
    """
    # Ensure image is RGB
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Ensure binary mask
    mask = (binary_mask > 0).astype(np.uint8)
    
    # Create colored overlay
    overlay = image_rgb.copy()
    overlay[mask == 1] = (
        alpha * np.array(color) + (1 - alpha) * overlay[mask == 1]
    ).astype(np.uint8)
    
    return overlay


def plot_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    diff_image: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    overlay: Optional[np.ndarray] = None,
    titles: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create comparison plot of change detection results
    
    Args:
        img1: Earlier image
        img2: Later image
        diff_image: Difference/change magnitude image
        predicted_mask: Predicted binary mask
        ground_truth: Optional ground truth mask
        overlay: Optional overlay image
        titles: Optional list of titles
        
    Returns:
        Matplotlib figure
    """
    if titles is None:
        titles = ["Earlier", "Later", "Difference", "Prediction"]
    
    # Determine number of subplots
    n_plots = 4
    if ground_truth is not None:
        n_plots += 1
    if overlay is not None:
        n_plots += 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Plot images
    axes[0].imshow(img1)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    axes[2].imshow(diff_image, cmap='hot')
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    
    axes[3].imshow(predicted_mask, cmap='gray')
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    
    idx = 4
    if ground_truth is not None:
        axes[idx].imshow(ground_truth, cmap='gray')
        axes[idx].set_title("Ground Truth")
        axes[idx].axis('off')
        idx += 1
    
    if overlay is not None:
        axes[idx].imshow(overlay)
        axes[idx].set_title("Overlay")
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["f1", "iou"]
) -> go.Figure:
    """
    Create bar chart comparing metrics across pipelines
    
    Args:
        metrics_dict: Dictionary mapping pipeline names to metric dicts
        metric_names: List of metrics to plot
        
    Returns:
        Plotly figure
    """
    pipeline_names = list(metrics_dict.keys())
    
    fig = go.Figure()
    
    for metric_name in metric_names:
        values = [metrics_dict[pipeline].get(metric_name, 0) for pipeline in pipeline_names]
        fig.add_trace(go.Bar(
            name=metric_name.upper(),
            x=pipeline_names,
            y=values,
        ))
    
    fig.update_layout(
        barmode='group',
        title="Pipeline Comparison",
        xaxis_title="Pipeline",
        yaxis_title="Score",
        yaxis_range=[0, 1]
    )
    
    return fig


def plot_metrics_line(
    metrics_history: List[Dict[str, float]],
    pipeline_name: str
) -> go.Figure:
    """
    Create line chart showing metrics over iterations
    
    Args:
        metrics_history: List of metric dictionaries
        pipeline_name: Name of the pipeline
        
    Returns:
        Plotly figure
    """
    iterations = list(range(len(metrics_history)))
    
    fig = go.Figure()
    
    for metric_name in ["precision", "recall", "f1", "iou"]:
        values = [m.get(metric_name, 0) for m in metrics_history]
        fig.add_trace(go.Scatter(
            x=iterations,
            y=values,
            name=metric_name.capitalize(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=f"{pipeline_name} Metrics Over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Score",
        yaxis_range=[0, 1]
    )
    
    return fig
