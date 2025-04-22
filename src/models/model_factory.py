# src/models/model_factory.py
"""
Factory functions for creating and configuring models.

This module provides factory functions to create various types of models,
including vision transformers, text models, and multimodal models with
appropriate configuration.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, Tuple

from ..models.vision.vision_transformer import VisionTransformer
from ..models.transformer import EncoderDecoderTransformer
from .multimodal.multimodal_integration import (
    MultiModalTransformer,
    CrossAttentionMultiModalTransformer,
)
from ..models.pretrained.huggingface_wrapper import (
    HuggingFaceTextModelWrapper,
    DimensionMatchingWrapper,
)
from ..utils.model_utils import count_parameters

logger = logging.getLogger(__name__)


def create_multimodal_model(args: Any, device: torch.device) -> nn.Module:
    """
    Create multimodal model with vision and text components.

    This factory function creates a multimodal model by configuring and
    instantiating vision and text models (either pretrained or from scratch)
    and combining them into a multimodal transformer.

    Args:
        args: Arguments containing model configuration
        device: Device to place the model on

    Returns:
        Configured multimodal model
    """
    # Check for model size presets for dimension matching
    if hasattr(args, "model_size") and args.model_size is not None:
        system_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        is_mps = system_device.type == "mps"

        if (
            args.model_size == "large"
            or args.model_size == "small"
            or args.model_size == "medium"
        ):  # Use 768 dimensions for all presets
            logger.info("Using 768-dimension models")
            args.use_pretrained_text = True
            args.vision_model = "vit-base"  # Native 768-dim
            args.fusion_dim = 768

            if is_mps:
                # MPS-compatible alternative for Apple Silicon
                args.text_model = "albert-base-v2"
                logger.info(
                    "Selected MPS-compatible models: vit-base (768) + albert-base-v2 (768)"
                )
            else:
                args.text_model = "bert-base-uncased"
                logger.info(
                    "Selected standard models: vit-base (768) + bert-base-uncased (768)"
                )

    logger.info("Creating multimodal model with pretrained components...")

    # Create vision transformer with pretrained weights
    try:
        import timm

        if args.use_pretrained:
            logger.info(f"Loading pretrained vision model: {args.vision_model}")
        else:
            logger.info(
                f"Creating vision model (without pretraining): {args.vision_model}"
            )

        # Check text model dimension first to determine appropriate vision model
        text_model_dim = 0
        if args.use_pretrained_text:
            # MobileBERT has 512 dim, BERT-base has 768 dim, MiniLM-384 has 384 dim
            if args.text_model == "mobilebert":
                text_model_dim = 512
            elif args.text_model in ["bert-base", "roberta-base"]:
                text_model_dim = 768
            elif args.text_model == "distilbert-base":
                text_model_dim = 768
            elif args.text_model == "albert-base":
                text_model_dim = 768
            elif args.text_model == "minilm-384":
                text_model_dim = 384

        # Keep things simple - use standard timm models
        # For 768 dimensions, use vit_base_patch16_224 which is standard and well-supported
        if "vit-base" in args.vision_model or args.fusion_dim == 768:
            # Standard ViT-base model with 768 dims
            logger.info(f"Loading standard ViT-base model with 768 dimensions")
            vision_model = timm.create_model(
                "vit_base_patch16_224", pretrained=args.use_pretrained
            )
            # Remove classification head
            vision_model.head = nn.Identity()

            # Get actual dimension from the model
            if hasattr(vision_model, "num_features"):
                vision_dim = vision_model.num_features
                logger.info(f"Detected vision dimension: {vision_dim}")
            else:
                # Fallback to standard ViT base dimension
                vision_dim = 768
                logger.info(f"Using standard ViT base dimension: {vision_dim}")
        else:
            # Generic fallback if a different model is specified
            logger.warning(
                f"Using fallback vision model because {args.vision_model} was specified"
            )
            # Default to ViT-base for reliability
            model_name = "vit_base_patch16_224"
            expected_dim = 768

            # Create model
            vision_model = timm.create_model(model_name, pretrained=args.use_pretrained)
            vision_model.head = nn.Identity()

            # Get dimension
            if hasattr(vision_model, "num_features"):
                vision_dim = vision_model.num_features
            else:
                vision_dim = expected_dim

            logger.info(f"Created fallback vision model with dimension: {vision_dim}")

    except ImportError:
        logger.warning("timm library not found, using standard vision transformer")
        # Fallback to standard implementation
        vision_config = {
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "num_classes": 1000,
            "embed_dim": 512,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }
        vision_model = VisionTransformer(**vision_config)
        vision_dim = vision_config["embed_dim"]

    # Create text transformer with pretrained weights
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading pretrained text model: {args.text_model}")

        # Check if we should use a pretrained text model from HuggingFace
        if args.use_pretrained_text:
            logger.info(
                f"Loading pretrained text model from HuggingFace: {args.text_model}"
            )

            # Use exact HuggingFace model name if provided directly
            if (
                "/" in args.text_model
                or args.text_model.startswith("microsoft")
                or args.text_model.startswith("google")
            ):
                # Direct HuggingFace model reference
                huggingface_model_name = args.text_model
                logger.info(f"Using exact HuggingFace model: {huggingface_model_name}")
            else:
                # Map model identifier to HuggingFace model name
                system_device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                is_mps = system_device.type == "mps"

                # Handle specific model size presets
                if args.model_size == "small":  # 384 dimensions
                    huggingface_model_name = "microsoft/MiniLM-L12-H384-uncased"
                    logger.info(f"Using 384-dim text model: {huggingface_model_name}")

                elif args.model_size == "medium":  # 512 dimensions
                    huggingface_model_name = "flaubert-small-cased"
                    logger.info(f"Using 512-dim text model: {huggingface_model_name}")

                elif args.model_size == "large":  # 768 dimensions
                    if is_mps:
                        huggingface_model_name = (
                            "albert-base-v2"  # MPS-friendly 768-dim model
                        )
                    else:
                        huggingface_model_name = "bert-base-uncased"
                    logger.info(f"Using 768-dim text model: {huggingface_model_name}")

                # If no model_size preset, use MPS-friendly models if on MPS
                elif is_mps:
                    if (
                        args.text_model == "bert-base"
                        or args.text_model == "bert-base-uncased"
                    ):
                        huggingface_model_name = (
                            "google/mobilebert-uncased"  # MPS-friendly alternative
                        )
                        logger.warning(
                            "⚠️ Automatically switched from BERT-base to MobileBERT for MPS compatibility"
                        )
                    elif args.text_model == "roberta-base":
                        huggingface_model_name = "distilroberta-base"  # Smaller model for better MPS compatibility
                        logger.warning(
                            "⚠️ Automatically switched from RoBERTa-base to DistilRoBERTa for MPS compatibility"
                        )
                    elif args.text_model == "distilbert-base":
                        huggingface_model_name = "distilbert-base-uncased"
                    elif args.text_model == "mobilebert":
                        huggingface_model_name = "google/mobilebert-uncased"
                    elif args.text_model == "albert-base":
                        huggingface_model_name = "albert-base-v2"
                    elif args.text_model == "minilm-384":
                        huggingface_model_name = "microsoft/MiniLM-L12-H384-uncased"
                    elif args.text_model == "flaubert-small-cased":
                        huggingface_model_name = "flaubert-small-cased"
                    else:
                        # Default to MobileBERT for MPS as safe option
                        huggingface_model_name = "google/mobilebert-uncased"
                        logger.info(
                            f"Defaulting to MobileBERT for unknown model on MPS: {args.text_model}"
                        )
                else:
                    # Standard model mapping for CPU/CUDA
                    if (
                        args.text_model == "bert-base"
                        or args.text_model == "bert-base-uncased"
                    ):
                        huggingface_model_name = "bert-base-uncased"
                    elif args.text_model == "roberta-base":
                        huggingface_model_name = "roberta-base"
                    elif args.text_model == "distilbert-base":
                        huggingface_model_name = "distilbert-base-uncased"
                    elif args.text_model == "mobilebert":
                        huggingface_model_name = "google/mobilebert-uncased"
                    elif args.text_model == "albert-base":
                        huggingface_model_name = "albert-base-v2"
                    elif args.text_model == "minilm-384":
                        huggingface_model_name = "microsoft/MiniLM-L12-H384-uncased"
                    elif args.text_model == "flaubert-small-cased":
                        huggingface_model_name = "flaubert-small-cased"
                    else:
                        # Default to BERT-base for other cases
                        huggingface_model_name = "bert-base-uncased"
                        logger.info(
                            f"Defaulting to BERT-base for unknown model: {args.text_model}"
                        )

            # Create model
            text_model = HuggingFaceTextModelWrapper(huggingface_model_name)
            text_dim = text_model.d_model

        else:
            # Not using pretrained - create standard transformer
            logger.info(
                f"Creating custom text transformer (without pretraining): {args.text_model}"
            )

            if args.text_model == "transformer-base":
                # Create larger custom transformer
                text_config = {
                    "src_vocab_size": 50000,
                    "tgt_vocab_size": 50000,
                    "d_model": 512,  # Match BERT dimension
                    "num_heads": 8,  # 768 is divisible by 12
                    "num_encoder_layers": 6,
                    "num_decoder_layers": 6,
                    "d_ff": 3072,
                    "dropout": 0.1,
                }
            else:
                # Use transformer-small
                logger.info("Using transformer-small configuration (384 dimensions)")
                text_config = {
                    "src_vocab_size": 50000,
                    "tgt_vocab_size": 50000,
                    "d_model": 384,  # Match vit-small dimension exactly (384 is divisible by 8)
                    "num_heads": 8,
                    "num_encoder_layers": 6,
                    "num_decoder_layers": 6,
                    "d_ff": 1536,  # 4 * d_model = 4 * 384 = 1536
                    "dropout": 0.1,
                }

            # Create model
            text_model = EncoderDecoderTransformer(**text_config)
            text_dim = text_config["d_model"]

    except ImportError:
        logger.warning(
            "transformers library not found, using standard text transformer"
        )
        # Fallback to standard implementation
        text_config = {
            "src_vocab_size": 50000,
            "tgt_vocab_size": 50000,
            "d_model": 512,  # 512 is divisible by 8
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1,
        }
        text_model = EncoderDecoderTransformer(**text_config)
        text_dim = text_config["d_model"]

    # Log parameter counts
    vision_params = count_parameters(vision_model)
    text_params = count_parameters(text_model)
    logger.info(f"Vision transformer parameters: {vision_params:,}")
    logger.info(f"Text transformer parameters: {text_params:,}")

    # Create multimodal model with custom initialization
    logger.info(
        f"Creating enhanced multimodal transformer with {args.fusion_type} fusion"
    )

    # Set fusion_dim to match the model dimensions if specified
    # Otherwise, use the default from args
    if args.fusion_dim is None:
        # Default to the vision dimension if fusion_dim not specified
        fusion_dim = vision_dim
    else:
        fusion_dim = args.fusion_dim

    # Log dimension information for debugging
    logger.info(
        f"Model dimensions - Vision: {vision_dim}, Text: {text_dim}, Fusion: {fusion_dim}"
    )

    # Check for dimension mismatch between vision and text models
    if vision_dim != text_dim:
        logger.warning(
            f"Dimension mismatch detected between vision ({vision_dim}) and text ({text_dim}) models"
        )

        # Create a dimension alignment layer to ensure they match before fusion
        if (
            args.text_model == "transformer-base"
            or args.text_model == "transformer-small"
        ):
            # For custom transformers, we can adjust the transformer itself
            logger.info(
                f"Setting text model dimension to match vision model: {vision_dim}"
            )
            # Update the model's dimension
            text_model.d_model = vision_dim
            text_dim = vision_dim
        else:
            # For pretrained models, add a projection layer
            logger.info(
                f"Adding explicit projection layer to match dimensions: {text_dim} → {vision_dim}"
            )

            # Wrap the text model with a projection
            text_model = DimensionMatchingWrapper(text_model, text_dim, vision_dim)
            # Move to the same device as the text model
            device = next(text_model.base_model.parameters()).device
            text_model.projection = text_model.projection.to(device)
            text_dim = vision_dim
            logger.info(f"Projection layer created on device: {device}")

    # After alignment, dimensions should match
    # Now adjust fusion_dim to match the aligned model dimensions if needed
    print(
        f"DIMENSION CHECK - Vision: {vision_dim}, Text: {text_dim}, Fusion: {fusion_dim}"
    )

    # Get actual text model dimension from HuggingFace if available
    if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "config"):
        hf_dim = getattr(text_model.encoder.config, "hidden_size", None)
        if hf_dim is not None:
            print(f"HuggingFace model hidden size: {hf_dim}")
            # If there's a mismatch, update text_dim
            if hf_dim != text_dim:
                print(
                    f"WARNING: Text dimension mismatch - Detected: {text_dim}, Actual: {hf_dim}"
                )
                text_dim = hf_dim

    if fusion_dim != vision_dim or fusion_dim != text_dim:
        # Prioritize text model dim for HuggingFace models, vision otherwise
        target_dim = text_dim if args.use_pretrained_text else vision_dim

        logger.warning(
            f"Dimension mismatch detected: Vision dim: {vision_dim}, Text dim: {text_dim}, Fusion dim: {fusion_dim}"
        )
        logger.info(
            f"Adjusting fusion_dim to match model dimensions: {fusion_dim} → {target_dim}"
        )
        fusion_dim = target_dim

        # CRITICAL: Update the args object as well so other components can access the updated dimension
        args.fusion_dim = fusion_dim
        logger.info(f"Updated args.fusion_dim to {args.fusion_dim}")

    logger.info(
        f"Aligned dimensions - Vision: {vision_dim}, Text: {text_dim}, Fusion: {fusion_dim}"
    )

    multimodal_model = CrossAttentionMultiModalTransformer(
        vision_model=vision_model,
        text_model=text_model,
        fusion_dim=fusion_dim,
        num_fusion_layers=3,  # Increased from 2 for better fusion
        num_heads=8,
        dropout=0.1,
        fusion_type=args.fusion_type,
    )

    # Set up parameter freezing based on argument
    if args.freeze_base_models:
        # Freeze base models initially (will be unfrozen during multi-stage training)
        for name, param in multimodal_model.named_parameters():
            if "vision_model" in name or "text_model" in name:
                param.requires_grad = False
        logger.info("Base models frozen for initial training phase")
    else:
        # Keep all parameters trainable for full fine-tuning
        for param in multimodal_model.parameters():
            param.requires_grad = True
        logger.info("All model parameters are trainable (including base models)")

    # Calculate trainable parameters
    trainable_params = sum(
        p.numel() for p in multimodal_model.parameters() if p.requires_grad
    )
    total_params = count_parameters(multimodal_model)
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}% of total)"
    )

    # Move model to device - ensure all components are on the same device
    multimodal_model = multimodal_model.to(device)

    # Explicitly check and move all submodules to ensure consistency
    multimodal_model.vision_model = multimodal_model.vision_model.to(device)
    multimodal_model.text_model = multimodal_model.text_model.to(device)
    multimodal_model.fusion_module = multimodal_model.fusion_module.to(device)
    if hasattr(multimodal_model, "classifier"):
        multimodal_model.classifier = multimodal_model.classifier.to(device)

    # Print device confirmation
    print(f"Model components successfully moved to {device}")
    print(f"- Vision model: {next(multimodal_model.vision_model.parameters()).device}")
    print(f"- Text model: {next(multimodal_model.text_model.parameters()).device}")
    print(
        f"- Fusion module: {next(multimodal_model.fusion_module.parameters()).device}"
    )

    return multimodal_model


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Factory functions for creating and configuring different types of models",
        "key_functions": [
            {
                "name": "create_multimodal_model",
                "signature": "create_multimodal_model(args: Any, device: torch.device) -> nn.Module",
                "brief_description": "Create a multimodal model with vision and text components",
            }
        ],
        "external_dependencies": ["torch", "timm", "transformers"],
        "complexity_score": 8,  # High complexity due to handling multiple model types and device compatibility
    }
