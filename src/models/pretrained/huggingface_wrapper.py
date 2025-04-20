# src/models/pretrained/huggingface_wrapper.py
"""
HuggingFace model wrapper for compatibility with multimodal models.

This module provides wrappers and adapters for using HuggingFace models
within the multimodal architecture, ensuring compatibility with the expected
interfaces and device handling.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Union, Tuple, Any, List

logger = logging.getLogger(__name__)


class HuggingFaceTextModelWrapper(nn.Module):
    """Wrapper for HuggingFace models to provide compatible interface."""

    def __init__(self, model_name: str):
        """
        Initialize the HuggingFace model wrapper.

        Args:
            model_name: Name of the HuggingFace model to load
        """
        super().__init__()

        # Get system device
        system_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Map to MPS-friendly models when needed
        # if (
        #     system_device.type == "mps"
        #     and "bert-base" in model_name.lower()
        # ):
        #     print(
        #         "⚠️ Detected MPS device - using MobileBERT instead of BERT for better compatibility"
        #     )
        #     model_name = "google/mobilebert-uncased"

        # Load the model on appropriate device
        if "mobilebert" in model_name.lower():
            from transformers import MobileBertModel, MobileBertTokenizer

            print(f"Loading MobileBERT model: {model_name}")
            self.encoder = MobileBertModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "mobilebert"
            print(f"MobileBERT hidden size: {self.d_model}")  # Diagnostic logging
            
        elif "albert" in model_name.lower():
            from transformers import AlbertModel, AlbertTokenizer

            print(f"Loading ALBERT model: {model_name}")
            self.encoder = AlbertModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "albert"
            
        elif "minilm" in model_name.lower():
            from transformers import AutoModel

            print(f"Loading MiniLM model: {model_name}")
            self.encoder = AutoModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "minilm"
            print(f"MiniLM hidden size: {self.d_model}")  # Diagnostic logging
            
        elif "flaubert" in model_name.lower():
            from transformers import FlaubertModel

            print(f"Loading FlauBERT model: {model_name}")
            self.encoder = FlaubertModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "flaubert"
            print(f"FlauBERT hidden size: {self.d_model}")  # Diagnostic logging
            
        elif "bert" in model_name.lower() and "distil" not in model_name.lower():
            from transformers import BertModel, BertTokenizer

            print(f"Loading BERT model: {model_name}")
            self.encoder = BertModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "bert"
            
        elif "roberta" in model_name.lower():
            from transformers import RobertaModel, RobertaTokenizer

            print(f"Loading RoBERTa model: {model_name}")
            self.encoder = RobertaModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "roberta"
            
        elif "distilbert" in model_name.lower():
            from transformers import DistilBertModel, DistilBertTokenizer

            print(f"Loading DistilBERT model: {model_name}")
            self.encoder = DistilBertModel.from_pretrained(model_name)
            self.d_model = self.encoder.config.hidden_size
            self.encoder_type = "distilbert"
            
        else:
            # Attempt to load model using AutoModel for any other HuggingFace model
            try:
                from transformers import AutoModel
                print(f"Attempting to load model with AutoModel: {model_name}")
                self.encoder = AutoModel.from_pretrained(model_name)
                self.d_model = self.encoder.config.hidden_size
                self.encoder_type = "auto"
                print(f"Successfully loaded model with dimension: {self.d_model}")
            except Exception as e:
                print(f"Failed to load with AutoModel: {str(e)}")
                raise ValueError(f"Unsupported model: {model_name}")

        # Try moving to the system device if MPS-compatible
        try:
            self.encoder = self.encoder.to(system_device)
            print(f"Successfully moved {model_name} to {system_device}")
        except Exception as e:
            print(
                f"Could not move model to {system_device}, using CPU instead: {str(e)}"
            )
            self.encoder = self.encoder.to("cpu")

        logger.info(f"Loaded {model_name} with dimension {self.d_model}")

    def encode(self, src, src_mask=None):
        """
        Encode text using the HuggingFace model.

        Args:
            src: Input token indices
            src_mask: Attention mask for padding

        Returns:
            Encoded text features
        """
        # Get original device
        input_device = src.device

        # MPS compatibility mode - check if we're on MPS and handle specially
        is_mps = input_device.type == "mps" or torch.backends.mps.is_available()

        # Get encoder's current device
        encoder_device = next(self.encoder.parameters()).device

        # Try direct processing for all models first
        # The extract_text_features method in multimodal_integration.py
        # will handle the CPU fallback if needed
        use_cpu_path = False

        # Define a CPU fallback function for reuse
        def process_on_cpu():
            print(
                f"Processing {getattr(self, 'encoder_type', 'model')} on CPU for compatibility"
            )
            # Move everything to CPU
            cpu_src = src.to("cpu")
            cpu_mask = src_mask.to("cpu") if src_mask is not None else None

            # Format mask if needed
            if cpu_mask is not None:
                if cpu_mask.dim() > 2:
                    cpu_mask = cpu_mask.squeeze(1)
                    if cpu_mask.dim() > 2:
                        cpu_mask = cpu_mask.squeeze(1)

            # Move encoder to CPU temporarily
            original_device = next(self.encoder.parameters()).device
            cpu_encoder = self.encoder.to("cpu")

            # Handle out-of-range indices if needed
            if hasattr(cpu_encoder, "embeddings") and hasattr(
                cpu_encoder.embeddings, "word_embeddings"
            ):
                vocab_size = cpu_encoder.embeddings.word_embeddings.weight.size(0)
                if torch.max(cpu_src) >= vocab_size:
                    cpu_src = torch.clamp(cpu_src, max=vocab_size - 1)

            try:
                # Process on CPU
                with torch.no_grad():
                    outputs = cpu_encoder(input_ids=cpu_src, attention_mask=cpu_mask)

                # Move encoder back
                self.encoder = self.encoder.to(original_device)

                # Return results on input device
                return outputs.last_hidden_state.to(input_device)
            except Exception as cpu_err:
                print(f"CPU fallback processing failed: {str(cpu_err)}")
                self.encoder = self.encoder.to(original_device)

                # Return zeros as last resort
                batch_size, seq_length = src.shape
                return torch.zeros(
                    batch_size,
                    seq_length,
                    self.d_model,
                    device=input_device,
                )

        # If we've decided to use CPU path, do it now
        if use_cpu_path:
            return process_on_cpu()

        # Otherwise, try normal processing
        try:
            # Format attention mask if needed
            if src_mask is not None:
                if src_mask.dim() > 2:
                    attention_mask = src_mask.squeeze(1)
                    if attention_mask.dim() > 2:
                        attention_mask = attention_mask.squeeze(1)
                else:
                    attention_mask = src_mask

                # Move mask to encoder device
                attention_mask = attention_mask.to(encoder_device)
            else:
                attention_mask = None

            # Move input to encoder device
            input_ids = src.to(encoder_device)

            # Handle out-of-range indices (important for BERT)
            if hasattr(self.encoder, "embeddings") and hasattr(
                self.encoder.embeddings, "word_embeddings"
            ):
                vocab_size = self.encoder.embeddings.word_embeddings.weight.size(0)
                # Check if indices are in range
                if torch.max(input_ids) >= vocab_size:
                    print(
                        f"Warning: Found input indices larger than vocabulary size ({vocab_size}). Clipping."
                    )
                    # Clip indices to valid range
                    input_ids = torch.clamp(input_ids, max=vocab_size - 1)

            # Regular processing with device alignment
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # Move result back to original device
                return outputs.last_hidden_state.to(input_device)

        except Exception as e:
            print(f"Error in HuggingFace text encoding: {str(e)}")
            print(f"Encoder type: {getattr(self, 'encoder_type', 'unknown')}")
            print(f"Encoder device: {encoder_device}, Input device: {input_device}")

            # Try CPU fallback
            try:
                return process_on_cpu()
            except Exception as fallback_err:
                print(f"All fallback attempts failed: {str(fallback_err)}")

            # Final emergency fallback - generate features with correct shape
            batch_size, seq_length = src.shape
            print(
                f"Using final fallback: zeros in correct shape on device {input_device}"
            )
            return torch.zeros(
                batch_size, seq_length, self.d_model, device=input_device
            )

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Forward pass - just calls encode for encoder-only models.

        Args:
            src: Input token indices
            tgt: Target token indices (not used for encoder-only models)
            src_mask: Attention mask for padding in source
            tgt_mask: Attention mask for padding in target (not used)

        Returns:
            Encoded text features
        """
        return self.encode(src, src_mask)


class DimensionMatchingWrapper(nn.Module):
    """
    Wrapper that adds projection layer to match dimensions between models.

    This is used when there's a dimension mismatch between vision and text models.
    """

    def __init__(self, base_model: nn.Module, input_dim: int, output_dim: int):
        """
        Initialize the dimension matching wrapper.

        Args:
            base_model: The model to wrap
            input_dim: Original dimension
            output_dim: Target dimension to project to
        """
        super().__init__()
        self.base_model = base_model
        self.projection = nn.Linear(input_dim, output_dim)
        self.d_model = output_dim  # Update dimension

        # Move projection to same device as base model initially
        base_device = next(base_model.parameters()).device
        self.projection = self.projection.to(base_device)

    def encode(self, *args, **kwargs):
        """
        Encode inputs and project to the target dimension.

        Args:
            *args: Positional arguments to pass to base model
            **kwargs: Keyword arguments to pass to base model

        Returns:
            Projected features
        """
        # Track original device for consistent return
        original_device = None
        if "src" in kwargs:
            original_device = kwargs["src"].device
        elif len(args) > 0:
            original_device = args[0].device

        # Get device from base model for consistency
        base_device = next(self.base_model.parameters()).device

        # Move inputs to base model device if needed
        if "src" in kwargs and kwargs["src"].device != base_device:
            kwargs["src"] = kwargs["src"].to(base_device)
        if (
            "src_mask" in kwargs
            and kwargs["src_mask"] is not None
            and kwargs["src_mask"].device != base_device
        ):
            kwargs["src_mask"] = kwargs["src_mask"].to(base_device)
        if len(args) > 0 and args[0].device != base_device:
            args = list(args)
            args[0] = args[0].to(base_device)
            args = tuple(args)

        # Get output from base model (now everything is on same device)
        base_output = self.base_model.encode(*args, **kwargs)

        # Ensure projection is on same device
        proj_device = next(self.projection.parameters()).device
        if base_output.device != proj_device:
            self.projection = self.projection.to(base_output.device)

        # Project to new dimension
        projected = self.projection(base_output)

        # Return to original device if requested
        if original_device is not None and projected.device != original_device:
            projected = projected.to(original_device)

        return projected

    def forward(self, *args, **kwargs):
        """Forward pass that calls encode."""
        return self.encode(*args, **kwargs)
