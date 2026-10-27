"""Change detection algorithms"""

from .differencing import absolute_difference
from .thresholding import otsu_threshold, fixed_threshold
from .morphology import morphological_opening, morphological_closing
from .cva import change_vector_analysis
from .ir_mad import ir_mad
from .pipelines import Pipeline, BaselinePipeline, CVAPipeline, IRMADPipeline

__all__ = [
    "absolute_difference",
    "otsu_threshold",
    "fixed_threshold",
    "morphological_opening",
    "morphological_closing",
    "change_vector_analysis",
    "ir_mad",
    "Pipeline",
    "BaselinePipeline",
    "CVAPipeline",
    "IRMADPipeline",
]
