"""
Multimodal models for integrating different data modalities.

This package contains models for processing and integrating different types of data,
such as text, images, and other modalities, using attention mechanisms and fusion techniques.
"""

from .cross_modal_attention_base import CrossModalAttention
from .gated_cross_modal_attention import GatedCrossModalAttention
from .bidirectional_cross_attention import BidirectionalCrossAttention
from .co_attention_fusion import CoAttentionFusion
from .multimodal_integration import (
    MultiModalTransformer,
    CrossAttentionMultiModalTransformer,
)
from .dual_encoder import DualEncoder
from .clip_style_direct_projection import CLIPStyleDirectProjection
from .multimodal_decoder_generation import MultimodalDecoderGeneration
from .vicreg_multimodal_model import VICRegMultimodalModel

__all__ = [
    "CrossModalAttention",
    "GatedCrossModalAttention",
    "BidirectionalCrossAttention",
    "CoAttentionFusion",
    "MultiModalTransformer",
    "CrossAttentionMultiModalTransformer",
    "DualEncoder",
    "CLIPStyleDirectProjection",
    "MultimodalDecoderGeneration",
    "VICRegMultimodalModel",
]
