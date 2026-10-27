"""Change detection pipelines combining multiple algorithms"""

import numpy as np
from typing import Dict, Any, Optional
from .differencing import absolute_difference
from .thresholding import otsu_threshold, fixed_threshold
from .morphology import morphological_refinement
from .cva import change_vector_analysis
from .ir_mad import ir_mad


class Pipeline:
    """Base class for change detection pipelines"""
    
    def __init__(self, name: str):
        self.name = name
        self.intermediate_results = {}
    
    def run(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        **params
    ) -> tuple:
        """
        Run the pipeline
        
        Args:
            img1: First image (earlier)
            img2: Second image (later)
            **params: Pipeline-specific parameters
            
        Returns:
            Tuple of (binary_mask, intermediate_results)
        """
        raise NotImplementedError


class BaselinePipeline(Pipeline):
    """Baseline pipeline: Absolute Difference → Otsu Threshold → Morphology"""
    
    def __init__(self):
        super().__init__("Baseline")
    
    def run(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        use_filter: bool = True,
        filter_type: str = "gaussian",
        filter_size: int = 5,
        threshold_method: str = "otsu",
        fixed_threshold_value: float = 0.5,
        morph_kernel_size: int = 3,
        morph_iterations: int = 1
    ) -> tuple:
        """
        Run baseline pipeline
        
        Args:
            img1: First image (earlier)
            img2: Second image (later)
            use_filter: Whether to apply smoothing filter
            filter_type: Type of filter ('gaussian' or 'median')
            filter_size: Size of filter kernel
            threshold_method: 'otsu' or 'fixed'
            fixed_threshold_value: Threshold value for fixed thresholding
            morph_kernel_size: Kernel size for morphological operations
            morph_iterations: Number of morphological iterations
            
        Returns:
            Tuple of (binary_mask, intermediate_results)
        """
        # Step 1: Absolute difference
        diff = absolute_difference(
            img1, img2,
            use_filter=use_filter,
            filter_type=filter_type,
            filter_size=filter_size
        )
        self.intermediate_results["difference"] = diff
        
        # Step 2: Thresholding
        if threshold_method == "otsu":
            threshold_val, binary = otsu_threshold(diff)
        else:
            binary = fixed_threshold(diff, fixed_threshold_value)
            threshold_val = fixed_threshold_value
        
        self.intermediate_results["threshold"] = threshold_val
        self.intermediate_results["thresholded"] = binary
        
        # Step 3: Morphological refinement
        refined = morphological_refinement(
            binary,
            kernel_size=morph_kernel_size,
            opening=True,
            closing=True,
            iterations=morph_iterations
        )
        self.intermediate_results["refined"] = refined
        
        return refined, self.intermediate_results


class CVAPipeline(Pipeline):
    """Change Vector Analysis pipeline: CVA → Threshold → Optional Morphology"""
    
    def __init__(self):
        super().__init__("CVA")
    
    def run(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold_method: str = "otsu",
        fixed_threshold_value: float = 0.3,
        use_morphology: bool = True,
        morph_kernel_size: int = 3,
        morph_iterations: int = 1
    ) -> tuple:
        """
        Run CVA pipeline
        
        Args:
            img1: First image (earlier)
            img2: Second image (later)
            threshold_method: 'otsu' or 'fixed'
            fixed_threshold_value: Threshold value for fixed thresholding
            use_morphology: Whether to apply morphological refinement
            morph_kernel_size: Kernel size for morphological operations
            morph_iterations: Number of morphological iterations
            
        Returns:
            Tuple of (binary_mask, intermediate_results)
        """
        # Step 1: Change Vector Analysis
        change_magnitude = change_vector_analysis(img1, img2)
        self.intermediate_results["change_magnitude"] = change_magnitude
        
        # Step 2: Thresholding
        if threshold_method == "otsu":
            threshold_val, binary = otsu_threshold(change_magnitude)
        else:
            binary = fixed_threshold(change_magnitude, fixed_threshold_value)
            threshold_val = fixed_threshold_value
        
        self.intermediate_results["threshold"] = threshold_val
        self.intermediate_results["thresholded"] = binary
        
        # Step 3: Optional morphological refinement
        if use_morphology:
            refined = morphological_refinement(
                binary,
                kernel_size=morph_kernel_size,
                opening=True,
                closing=True,
                iterations=morph_iterations
            )
            self.intermediate_results["refined"] = refined
        else:
            refined = binary
        
        return refined, self.intermediate_results


class IRMADPipeline(Pipeline):
    """IR-MAD pipeline: IR-MAD → Threshold → Optional Morphology"""
    
    def __init__(self):
        super().__init__("IR-MAD")
    
    def run(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        max_iterations: int = 30,
        convergence_threshold: float = 1e-6,
        threshold_method: str = "otsu",
        fixed_threshold_value: float = 0.95,
        use_morphology: bool = True,
        morph_kernel_size: int = 3,
        morph_iterations: int = 1
    ) -> tuple:
        """
        Run IR-MAD pipeline
        
        Args:
            img1: First image (earlier)
            img2: Second image (later)
            max_iterations: Maximum iterations for IR-MAD
            convergence_threshold: Convergence threshold for IR-MAD
            threshold_method: 'otsu' or 'fixed'
            fixed_threshold_value: Threshold value for fixed thresholding
            use_morphology: Whether to apply morphological refinement
            morph_kernel_size: Kernel size for morphological operations
            morph_iterations: Number of morphological iterations
            
        Returns:
            Tuple of (binary_mask, intermediate_results)
        """
        # Step 1: IR-MAD
        change_prob, mad_magnitude, weights = ir_mad(
            img1, img2,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        self.intermediate_results["change_probability"] = change_prob
        self.intermediate_results["mad_magnitude"] = mad_magnitude
        self.intermediate_results["weights"] = weights
        
        # Step 2: Thresholding
        if threshold_method == "otsu":
            threshold_val, binary = otsu_threshold(change_prob)
        else:
            binary = fixed_threshold(change_prob, fixed_threshold_value)
            threshold_val = fixed_threshold_value
        
        self.intermediate_results["threshold"] = threshold_val
        self.intermediate_results["thresholded"] = binary
        
        # Step 3: Optional morphological refinement
        if use_morphology:
            refined = morphological_refinement(
                binary,
                kernel_size=morph_kernel_size,
                opening=True,
                closing=True,
                iterations=morph_iterations
            )
            self.intermediate_results["refined"] = refined
        else:
            refined = binary
        
        return refined, self.intermediate_results


def get_pipeline(pipeline_name: str) -> Pipeline:
    """
    Get a pipeline by name
    
    Args:
        pipeline_name: Name of pipeline ('baseline', 'cva', 'ir_mad')
        
    Returns:
        Pipeline instance
    """
    pipelines = {
        "baseline": BaselinePipeline(),
        "cva": CVAPipeline(),
        "ir_mad": IRMADPipeline(),
    }
    
    pipeline_name_lower = pipeline_name.lower()
    if pipeline_name_lower not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(pipelines.keys())}")
    
    return pipelines[pipeline_name_lower]
