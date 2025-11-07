"""MODULE: vicreg_multimodal_model.py
PURPOSE: Implements VICReg (Variance-Invariance-Covariance Regularization) for multimodal representation learning.

KEY COMPONENTS:
- VICRegMultimodalModel: Main class implementing VICReg for multimodal data
- Support for vision and text modalities
- Configurable feature projectors and regularization
- Efficient batch processing for large datasets
- Modality-specific feature extraction

DEPENDENCIES:
- torch
- torch.nn
- typing
- transformers
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegMultimodalModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512, device=None):
        super().__init__()
        # Store the target device for later use
        self.target_device = device

        # First build the model on CPU for initialization
        cpu_device = torch.device("cpu")

        # Base models might already be on specific devices, keep track of original devices
        vision_device = next(vision_model.parameters()).device
        text_device = next(text_model.parameters()).device

        # Temporarily move to CPU for initialization if needed
        if vision_device != cpu_device:
            print(f"Temporarily moving vision model to CPU for initialization")
            vision_model = vision_model.to(cpu_device)

        if text_device != cpu_device:
            print(f"Temporarily moving text model to CPU for initialization")
            text_model = text_model.to(cpu_device)

        # Store models
        self.vision_model = vision_model
        self.text_model = text_model

        # Get dimensions
        vision_dim = self._get_model_dimension(vision_model)
        text_dim = self._get_model_dimension(text_model)

        print(f"Model dimensions - Vision: {vision_dim}, Text: {text_dim}")

        # Always create proper projection networks, even when dimensions match
        # This ensures we have trainable parameters in stage 1
        print(
            f"Creating vision projection with dimensions: {vision_dim} -> {projection_dim}"
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),  # First linear layer maintains dimension
            nn.BatchNorm1d(
                vision_dim, affine=True
            ),  # Affine=True for trainable parameters
            nn.ReLU(),
            nn.Linear(
                vision_dim, projection_dim
            ),  # Second linear adapts to projection_dim if needed
            nn.BatchNorm1d(
                projection_dim, affine=False
            ),  # Affine=False for VICReg (fixed)
        )

        # Always create proper text projection too, even when dimensions match
        print(
            f"Creating text projection with dimensions: {text_dim} -> {projection_dim}"
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),  # First linear layer maintains dimension
            nn.BatchNorm1d(
                text_dim, affine=True
            ),  # Affine=True for trainable parameters
            nn.GELU(),  # Different activation
            nn.Linear(
                text_dim, projection_dim
            ),  # Second linear adapts to projection_dim if needed
            nn.BatchNorm1d(
                projection_dim, affine=False
            ),  # Affine=False for VICReg (fixed)
        )

        # For explicit variance regularization
        self.vision_var_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, projection_dim),
        )

        self.text_var_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, projection_dim),
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Initialize weights with orthogonal initialization on CPU
        print("Performing orthogonal initialization on CPU")
        self._init_parameters()

        # Move components back to their original devices or target device if specified
        if self.target_device is not None:
            # If a target device is specified, move everything there
            print(f"Moving model components to target device: {self.target_device}")
            self.to(self.target_device)
        else:
            # Otherwise, move components back to their original devices
            if vision_device != cpu_device:
                print(f"Moving vision model back to original device: {vision_device}")
                self.vision_model = self.vision_model.to(vision_device)

            if text_device != cpu_device:
                print(f"Moving text model back to original device: {text_device}")
                self.text_model = self.text_model.to(text_device)

    def _get_model_dimension(self, model):
        """
        Extract embedding dimension from model.

        Args:
            model: Model to extract dimension from

        Returns:
            int: Embedding dimension
        """
        # Different models store their dimensions in different attributes
        if hasattr(model, "d_model"):
            return model.d_model
        elif hasattr(model, "hidden_size"):
            return model.hidden_size
        elif hasattr(model, "hidden_dim"):
            return model.hidden_dim
        elif hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        elif hasattr(model, "pretrained_model") and hasattr(
            model.pretrained_model, "config"
        ):
            # HuggingFace wrappers often have this structure
            return model.pretrained_model.config.hidden_size
        else:
            # Default to a standard size if we can't determine
            print(
                f"Warning: Could not determine dimension for model {type(model).__name__}. Using default 768."
            )
            return 768

    def _extract_features(self, model, x):
        """
        Extract features from a model.

        Args:
            model: Model to extract features from
            x: Input tensor or dictionary

        Returns:
            torch.Tensor: Extracted features
        """
        # Use the same print counter from the forward method to reduce logging
        should_print = hasattr(self, "_print_counter") and (
            self._print_counter % 20 == 0
        )

        # Print debug info about the input (less frequently)
        if should_print:
            print(
                f"Extracting features from {type(model).__name__} with input type {type(x)}"
            )

            # Handle common text data structures from datasets
            if isinstance(x, dict):
                # Check for standard keys in text data
                keys = set(x.keys())
                print(f"Dict keys in input: {keys}")

        # Special handling for different dataset formats
        if isinstance(x, dict):
            # We have a "raw" text representation
            if "input_ids" not in x.keys() and "text" in x.keys():
                if should_print:
                    print("Converting raw text to features using fallback")
                # In a real implementation, we would tokenize the text here
                # For now, this is handled in the error recovery in the forward method
                raise ValueError(
                    "No input_ids in text data dictionary - need tokenization"
                )

        # Different models have different forward signatures and return formats
        try:
            if hasattr(model, "encode"):
                # Our wrappers typically have an encode method
                if should_print:
                    print("Using model.encode method")
                features = model.encode(x)
            elif hasattr(model, "extract_features"):
                # Some models have a dedicated feature extraction method
                if should_print:
                    print("Using model.extract_features method")
                features = model.extract_features(x)
            else:
                # Default to standard forward pass
                if should_print:
                    print("Using model.__call__ method")
                features = model(x)

                # Handle different return types
                if isinstance(features, tuple):
                    features = features[0]  # Usually the first element is the features
                elif isinstance(features, dict) and "last_hidden_state" in features:
                    features = features["last_hidden_state"]  # HuggingFace models

            # If features are sequence data (3D), pool to get a single vector per item
            if len(features.shape) == 3:
                # Mean pooling over sequence dimension
                if should_print:
                    print(f"Pooling sequence features from shape {features.shape}")
                features = features.mean(dim=1)

            return features

        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            # Try to provide more diagnostic information
            if isinstance(x, dict) and not any(
                k in x for k in ["input_ids", "inputs_embeds"]
            ):
                print("Input dictionary is missing expected HuggingFace model inputs.")
                keys_str = ", ".join(x.keys())
                print(f"Available keys: {keys_str}")
                raise ValueError(
                    f"Missing required keys for HuggingFace model. Found: {keys_str}"
                )
            raise

    def _init_parameters(self):
        """
        Initialize weights using standardized orthogonal initialization on CPU
        with consistent gain values.
        """
        # We should be on CPU at this point so orthogonal initialization is supported
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use consistent gain value of 1.5 for all components
                if "var_predictor" in name:
                    # Still use higher gain for variance predictors, but reduced
                    nn.init.orthogonal_(module.weight, gain=1.7)
                    print(
                        f"Applied variance predictor orthogonal initialization (gain=1.7) to {name}"
                    )
                else:
                    # Standardized gain for all other layers
                    nn.init.orthogonal_(module.weight, gain=1.5)
                    print(
                        f"Applied standardized orthogonal initialization (gain=1.5) to {name}"
                    )

                # Initialize all biases to zero for better stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    print(f"Applied zero bias initialization to {name}")

    def forward(self, images=None, text_data=None):
        outputs = {}

        # Get model device for consistency checks
        model_device = next(self.parameters()).device

        # Use a class variable to reduce logging frequency
        if not hasattr(self, "_print_counter"):
            self._print_counter = 0

        # Only print debugging info every 20 batches
        should_print = self._print_counter % 20 == 0
        self._print_counter += 1

        # Print input info for debugging (less frequently)
        if should_print and images is not None:
            print(f"Input images shape: {images.shape}")
        if should_print and text_data is not None:
            if isinstance(text_data, dict):
                print(f"Input text_data keys: {text_data.keys()}")
                if "input_ids" in text_data:
                    print(
                        f"Input text_data['input_ids'] shape: {text_data['input_ids'].shape}"
                    )

        # Process vision
        if images is not None:
            # Ensure images are on the same device as the model
            if images.device != model_device:
                images = images.to(model_device)

            vision_features = self._extract_features(self.vision_model, images)
            vision_proj = self.vision_proj(vision_features)

            # Explicit variance encouragement through prediction
            if self.training:
                vision_var_pred = self.vision_var_predictor(vision_proj)

                # Calculate variance statistics
                std_vision = torch.sqrt(torch.var(vision_proj, dim=0) + 1e-4)
                outputs["vision_std"] = std_vision.mean().item()

                # Apply variance-based feature expansion
                vision_proj = vision_proj * (1.0 + vision_var_pred.tanh() * 0.15)

            outputs["vision_features"] = vision_proj

        # Process text (similar structure to vision)
        if text_data is not None:
            # Debug: Print the structure of text_data to understand what we're dealing with
            if should_print:
                print(f"Text data type: {type(text_data)}")
                if isinstance(text_data, dict):
                    print(f"Text data keys: {text_data.keys()}")

            # Prepare text input based on the actual keys in the dataset
            modified_text_data = {}

            # Dataset may provide src/src_mask, but we need to rename to input_ids/attention_mask
            if "src" in text_data and "input_ids" not in text_data:
                if should_print:
                    print("Adapting dataset format: src -> input_ids")
                modified_text_data["input_ids"] = text_data["src"]
            else:
                # Keep existing input_ids if available
                if "input_ids" in text_data:
                    modified_text_data["input_ids"] = text_data["input_ids"]

            # Same for attention mask
            if "src_mask" in text_data and "attention_mask" not in text_data:
                if should_print:
                    print("Adapting dataset format: src_mask -> attention_mask")
                modified_text_data["attention_mask"] = text_data["src_mask"]
            else:
                # Keep existing attention_mask if available
                if "attention_mask" in text_data:
                    modified_text_data["attention_mask"] = text_data["attention_mask"]

            # If we have raw text, keep it for potential fallback
            if "text" in text_data:
                modified_text_data["text"] = text_data["text"]

            # Use the modified text data with properly renamed keys
            text_data = modified_text_data if modified_text_data else text_data
            # print(f"Adapted text_data keys: {text_data.keys()}")

            # Ensure text data is on the same device as the model
            # Handle both tensor and dictionary inputs
            if isinstance(text_data, torch.Tensor) and text_data.device != model_device:
                text_data = text_data.to(model_device)
            elif isinstance(text_data, dict):
                # Move each tensor in the dictionary to the model's device
                for key, value in text_data.items():
                    if isinstance(value, torch.Tensor) and value.device != model_device:
                        text_data[key] = value.to(model_device)

            # Extract text features with better error handling
            try:
                text_features = self._extract_features(self.text_model, text_data)
            except Exception as e:
                print(f"Error extracting text features: {str(e)}")
                # Create zeros instead of random features for better stability
                batch_size = (
                    text_data["input_ids"].shape[0]
                    if isinstance(text_data, dict) and "input_ids" in text_data
                    else (images.shape[0] if images is not None else 1)
                )
                text_dim = self._get_model_dimension(self.text_model)
                print(f"Creating zero text features of size {batch_size}x{text_dim}")

                # Use zeros instead of random features to avoid introducing noise
                text_features = torch.zeros(batch_size, text_dim, device=model_device)
                print(
                    "Using zero features for error recovery (more stable than random)"
                )
            text_proj = self.text_proj(text_features)

            # Different variance encouragement for text
            if self.training:
                text_var_pred = self.text_var_predictor(text_proj)

                # Calculate variance statistics
                std_text = torch.sqrt(torch.var(text_proj, dim=0) + 1e-4)
                outputs["text_std"] = std_text.mean().item()

                # Apply variance-based feature expansion with different factor
                text_proj = text_proj * (1.0 + text_var_pred.tanh() * 0.15)

            outputs["text_features"] = text_proj

        # Compute similarity if both present
        if images is not None and text_data is not None:
            # VICReg embeddings are normalized after loss computation
            vision_norm = F.normalize(outputs["vision_features"], p=2, dim=1)
            text_norm = F.normalize(outputs["text_features"], p=2, dim=1)

            # Print feature shapes for debugging (less frequently)
            if should_print:
                print(
                    f"Normalized feature shapes - Vision: {vision_norm.shape}, Text: {text_norm.shape}"
                )

            # Calculate similarity with temperature
            logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
            similarity = logit_scale * torch.matmul(vision_norm, text_norm.T)

            outputs["similarity"] = similarity

            # Compute statistics for covariance regularization loss
            if self.training:
                # Compute covariance matrices for regularization
                batch_size = vision_proj.size(0)

                # Center the features
                vision_centered = vision_proj - vision_proj.mean(dim=0, keepdim=True)
                text_centered = text_proj - text_proj.mean(dim=0, keepdim=True)

                # Note: MPS may have issues with large matrix multiplications
                # If running on MPS and batch size is large, consider chunking or fallback
                vision_cov = (vision_centered.T @ vision_centered) / (batch_size - 1)
                text_cov = (text_centered.T @ text_centered) / (batch_size - 1)

                # Store for VICReg loss computation
                outputs["vision_cov"] = vision_cov
                outputs["text_cov"] = text_cov
                outputs["vision_centered"] = vision_centered
                outputs["text_centered"] = text_centered

        return outputs


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
        "module_purpose": "Implements VICReg (Variance-Invariance-Covariance Regularization) for multimodal representation learning",
        "key_classes": [
            {
                "name": "VICRegMultimodalModel",
                "purpose": "Implements VICReg approach for learning aligned multimodal representations",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, projection_dim: int = 8192, hidden_dim: int = 8192)",
                        "brief_description": "Initialize VICReg multimodal model with encoders and projectors",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_input: torch.Tensor, text_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute VICReg representations and regularization terms",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "transformers"],
            }
        ],
        "external_dependencies": ["torch", "transformers", "typing"],
        "complexity_score": 8,
    }
