import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List, Any
import logging
import math

logger = logging.getLogger(__name__)

class MultimodalDecoderGeneration(nn.Module):
    """
    Multimodal Decoder Generation model.
    
    This model implements a generative architecture that takes inputs from multiple
    modalities (text and vision) and generates text outputs. It combines:
    1. A vision encoder for processing images
    2. A text encoder for processing text inputs
    3. A cross-modal fusion mechanism
    4. A text decoder for generating outputs
    
    The architecture supports conditioning the decoder on either or both modalities.
    """
    
    def __init__(
        self,
        vision_model: nn.Module,
        text_encoder: nn.Module,
        text_decoder: nn.Module,
        fusion_dim: int = 768,
        num_fusion_layers: int = 2,
        use_cross_attention: bool = True,
        max_sequence_length: int = 128,
        fusion_dropout: float = 0.1,
        tie_embeddings: bool = True,
        vocab_size: Optional[int] = None,
        use_gated_fusion: bool = True,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multimodal decoder generation model.
        
        Args:
            vision_model: Vision encoder model
            text_encoder: Text encoder model
            text_decoder: Text decoder model (typically transformer decoder)
            fusion_dim: Dimension of the fusion space
            num_fusion_layers: Number of fusion transformer layers
            use_cross_attention: Whether to use cross-attention for fusion
            max_sequence_length: Maximum sequence length for generation
            fusion_dropout: Dropout rate in fusion layers
            tie_embeddings: Whether to tie input/output embeddings for decoder
            vocab_size: Vocabulary size (if not tied to decoder's embedding)
            use_gated_fusion: Whether to use gated fusion mechanism
            generation_config: Configuration for text generation
        """
        super().__init__()
        self.vision_model = vision_model
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.max_sequence_length = max_sequence_length
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        
        # Default generation config
        self.generation_config = generation_config or {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "max_length": 128,
        }
        
        # Detect dimensions
        self.vision_dim = self._get_model_dimension(vision_model)
        self.text_encoder_dim = self._get_model_dimension(text_encoder)
        self.text_decoder_dim = self._get_model_dimension(text_decoder)
        
        # Create necessary projections for dimension matching
        self.vision_projection = nn.Linear(self.vision_dim, fusion_dim)
        self.text_encoder_projection = nn.Linear(self.text_encoder_dim, fusion_dim)
        
        # Create cross-modal fusion mechanism
        if use_cross_attention:
            self.cross_modal_fusion = CrossModalFusionTransformer(
                dim=fusion_dim,
                num_layers=num_fusion_layers,
                num_heads=8,
                dropout=fusion_dropout
            )
        else:
            # Simple fusion with gating
            if use_gated_fusion:
                self.gated_fusion = GatedMultimodalFusion(
                    vision_dim=fusion_dim,
                    text_dim=fusion_dim,
                    output_dim=fusion_dim,
                    dropout=fusion_dropout
                )
            else:
                # Simple addition with projection
                self.fusion_projection = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(fusion_dim, fusion_dim)
                )
        
        # Projection from fusion space to decoder input space
        self.fusion_to_decoder = nn.Linear(fusion_dim, self.text_decoder_dim)
        
        # Create output projection if not tying with decoder embeddings
        if not tie_embeddings:
            assert vocab_size is not None, "vocab_size must be provided when tie_embeddings=False"
            self.output_projection = nn.Linear(self.text_decoder_dim, vocab_size)
        else:
            self.output_projection = None
        
        # Initialize weights
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters for better training dynamics."""
        # Initialize projection layers
        for module in [self.vision_projection, self.text_encoder_projection, self.fusion_to_decoder]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_model_dimension(self, model: nn.Module) -> int:
        """
        Extract the output dimension from a model.
        
        Args:
            model: Neural network model
            
        Returns:
            The dimension of the model's output embeddings
        """
        # Try various common attribute names for dimensions
        if hasattr(model, "num_features"):
            return model.num_features
        elif hasattr(model, "embed_dim"):
            return model.embed_dim
        elif hasattr(model, "d_model"):
            return model.d_model
        elif hasattr(model, "hidden_size"):
            return model.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        
        # If no dimension found, use a common default but log a warning
        logger.warning(f"Could not determine dimension for model {type(model).__name__}, using default of 768")
        return 768
    
    def prepare_inputs_for_decoder(
        self,
        vision_features: Optional[torch.Tensor] = None,
        encoder_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare fused representation from modalities for the decoder.
        
        Args:
            vision_features: Features from vision encoder [batch_size, vision_dim] or [batch_size, seq_len, vision_dim]
            encoder_features: Features from text encoder [batch_size, seq_len, encoder_dim]
            attention_mask: Attention mask for the text encoder [batch_size, seq_len]
            
        Returns:
            Tuple of (fused_features, attention_mask)
        """
        batch_size = vision_features.shape[0] if vision_features is not None else encoder_features.shape[0]
        
        # Process vision features if provided
        if vision_features is not None:
            # Project to fusion dimension
            if len(vision_features.shape) == 2:
                # [batch_size, vision_dim] -> [batch_size, 1, fusion_dim]
                vision_projected = self.vision_projection(vision_features).unsqueeze(1)
            else:
                # [batch_size, seq_len, vision_dim] -> [batch_size, seq_len, fusion_dim]
                vision_projected = self.vision_projection(vision_features)
        else:
            # Create empty vision features
            vision_projected = torch.zeros(
                (batch_size, 1, self.fusion_dim),
                device=encoder_features.device if encoder_features is not None else self.vision_projection.weight.device
            )
        
        # Process encoder features if provided
        if encoder_features is not None:
            # Project to fusion dimension
            encoder_projected = self.text_encoder_projection(encoder_features)
        else:
            # Create empty encoder features
            encoder_projected = torch.zeros(
                (batch_size, 1, self.fusion_dim),
                device=vision_projected.device
            )
            attention_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=vision_projected.device)
        
        # Fuse the modalities
        if self.use_cross_attention:
            # Use cross-modal transformer for fusion
            fused_features = self.cross_modal_fusion(
                vision_features=vision_projected,
                text_features=encoder_projected,
                text_attention_mask=attention_mask
            )
        elif self.use_gated_fusion:
            # Use gated multimodal fusion
            # First pool encoder features if they're sequence data
            if len(encoder_projected.shape) == 3 and encoder_projected.shape[1] > 1:
                # Apply attention mask for pooling
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    encoder_pooled = (encoder_projected * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    encoder_pooled = encoder_projected.mean(dim=1)
            else:
                encoder_pooled = encoder_projected.squeeze(1)
            
            # Pool vision features if they're sequence data
            if len(vision_projected.shape) == 3 and vision_projected.shape[1] > 1:
                vision_pooled = vision_projected.mean(dim=1)
            else:
                vision_pooled = vision_projected.squeeze(1)
            
            # Apply gated fusion
            fused_pooled = self.gated_fusion(vision_pooled, encoder_pooled)
            
            # Expand to sequence for decoder
            fused_features = fused_pooled.unsqueeze(1)
        else:
            # Simple concatenation and projection
            # First pool the features if they're sequence data
            if len(encoder_projected.shape) == 3 and encoder_projected.shape[1] > 1:
                # Apply attention mask for pooling
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    encoder_pooled = (encoder_projected * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    encoder_pooled = encoder_projected.mean(dim=1)
            else:
                encoder_pooled = encoder_projected.squeeze(1)
            
            # Pool vision features if they're sequence data
            if len(vision_projected.shape) == 3 and vision_projected.shape[1] > 1:
                vision_pooled = vision_projected.mean(dim=1)
            else:
                vision_pooled = vision_projected.squeeze(1)
            
            # Concatenate and project
            concatenated = torch.cat([vision_pooled, encoder_pooled], dim=1)
            fused_pooled = self.fusion_projection(concatenated)
            
            # Expand to sequence for decoder
            fused_features = fused_pooled.unsqueeze(1)
        
        # Project to decoder dimension
        decoder_inputs = self.fusion_to_decoder(fused_features)
        
        # Create new attention mask for the fused feature sequence
        fused_attention_mask = torch.ones(
            (batch_size, decoder_inputs.shape[1]),
            dtype=torch.bool,
            device=decoder_inputs.device
        )
        
        return decoder_inputs, fused_attention_mask
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the multimodal decoder generation model.
        
        Args:
            images: Optional image inputs [batch_size, channels, height, width]
            encoder_input_ids: Text encoder input IDs [batch_size, seq_len]
            encoder_attention_mask: Text encoder attention mask [batch_size, seq_len]
            decoder_input_ids: Text decoder input IDs [batch_size, seq_len]
            decoder_attention_mask: Text decoder attention mask [batch_size, seq_len]
            labels: Optional labels for computing loss [batch_size, seq_len]
            
        Returns:
            Dictionary with logits, loss (if labels provided), and other outputs
        """
        outputs = {}
        
        # Process vision if provided
        if images is not None:
            # Get vision features
            vision_features = self.vision_model(images)
            
            # Handle different output formats
            if isinstance(vision_features, dict):
                if "last_hidden_state" in vision_features:
                    vision_features = vision_features["last_hidden_state"]
                elif "pooler_output" in vision_features:
                    vision_features = vision_features["pooler_output"]
            
            outputs["vision_features"] = vision_features
        else:
            vision_features = None
        
        # Process text encoder if provided
        if encoder_input_ids is not None:
            # Get encoder outputs
            encoder_outputs = self.text_encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )
            
            # Extract features from appropriate key
            if "last_hidden_state" in encoder_outputs:
                encoder_features = encoder_outputs["last_hidden_state"]
            else:
                # Use the first output if it's not a dictionary
                encoder_features = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs
            
            outputs["encoder_features"] = encoder_features
        else:
            encoder_features = None
            encoder_attention_mask = None
        
        # Prepare inputs for the decoder
        decoder_conditioning, decoder_conditioning_mask = self.prepare_inputs_for_decoder(
            vision_features=vision_features,
            encoder_features=encoder_features,
            attention_mask=encoder_attention_mask
        )
        
        # Process with the decoder
        if decoder_input_ids is not None:
            # For transformer decoder models that expect specific input format
            if hasattr(self.text_decoder, "model_parallel") or hasattr(self.text_decoder, "is_decoder"):
                # HuggingFace-style decoder
                decoder_outputs = self.text_decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=decoder_conditioning,
                    encoder_attention_mask=decoder_conditioning_mask,
                    return_dict=True
                )
                
                # Extract decoder features
                if hasattr(decoder_outputs, "last_hidden_state"):
                    decoder_features = decoder_outputs.last_hidden_state
                else:
                    decoder_features = decoder_outputs[0]
            else:
                # Generic decoder implementation
                decoder_features = self.text_decoder(
                    decoder_input_ids, 
                    decoder_conditioning,
                    decoder_attention_mask
                )
            
            # Project to vocabulary if needed
            if self.output_projection is not None:
                logits = self.output_projection(decoder_features)
            else:
                # Try to use decoder's output projection
                if hasattr(self.text_decoder, "lm_head"):
                    logits = self.text_decoder.lm_head(decoder_features)
                elif hasattr(self.text_decoder, "output_projection"):
                    logits = self.text_decoder.output_projection(decoder_features)
                else:
                    # If no projection is found, raise an error
                    raise ValueError(
                        "No output projection found. Either set tie_embeddings=False and provide vocab_size, "
                        "or use a decoder with a lm_head or output_projection attribute."
                    )
            
            outputs["logits"] = logits
            
            # Compute loss if labels provided
            if labels is not None:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                
                # Shift labels for autoregressive loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs["loss"] = loss
        
        return outputs
    
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text based on visual and/or textual inputs.
        
        Args:
            images: Optional image inputs [batch_size, channels, height, width]
            encoder_input_ids: Text encoder input IDs [batch_size, seq_len]
            encoder_attention_mask: Text encoder attention mask [batch_size, seq_len]
            generation_config: Optional configuration for generation
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        # Merge configurations
        config = {**self.generation_config, **(generation_config or {})}
        
        # Process vision if provided
        if images is not None:
            # Get vision features
            vision_features = self.vision_model(images)
            
            # Handle different output formats
            if isinstance(vision_features, dict):
                if "last_hidden_state" in vision_features:
                    vision_features = vision_features["last_hidden_state"]
                elif "pooler_output" in vision_features:
                    vision_features = vision_features["pooler_output"]
        else:
            vision_features = None
        
        # Process text encoder if provided
        if encoder_input_ids is not None:
            # Get encoder outputs
            encoder_outputs = self.text_encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )
            
            # Extract features from appropriate key
            if "last_hidden_state" in encoder_outputs:
                encoder_features = encoder_outputs["last_hidden_state"]
            else:
                # Use the first output if it's not a dictionary
                encoder_features = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs
        else:
            encoder_features = None
            encoder_attention_mask = None
        
        # Prepare inputs for the decoder
        decoder_conditioning, decoder_conditioning_mask = self.prepare_inputs_for_decoder(
            vision_features=vision_features,
            encoder_features=encoder_features,
            attention_mask=encoder_attention_mask
        )
        
        # Get the device from conditioning
        device = decoder_conditioning.device
        
        # Determine batch size
        batch_size = decoder_conditioning.shape[0]
        
        # Initialize with BOS token if provided in config
        if "bos_token_id" in config and config["bos_token_id"] is not None:
            input_ids = torch.full(
                (batch_size, 1),
                config["bos_token_id"],
                dtype=torch.long,
                device=device
            )
        else:
            # Default to 0 (common BOS token ID)
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        # Get model parameters
        temperature = config.get("temperature", 1.0)
        top_k = config.get("top_k", 0)
        top_p = config.get("top_p", 1.0)
        repetition_penalty = config.get("repetition_penalty", 1.0)
        no_repeat_ngram_size = config.get("no_repeat_ngram_size", 0)
        max_length = config.get("max_length", self.max_sequence_length)
        
        # Main generation loop
        for _ in range(max_length - 1):  # -1 because we already have one token
            # Create attention mask for decoder
            decoder_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            # Get logits for next token
            outputs = self.forward(
                encoder_input_ids=None,  # No need to reprocess text
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_conditioning=decoder_conditioning,
                decoder_conditioning_mask=decoder_conditioning_mask
            )
            
            # Extract logits for the next token prediction
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Create a mask of previously generated tokens
                prev_tokens = input_ids.tolist()
                for batch_idx in range(batch_size):
                    for prev_token in set(prev_tokens[batch_idx]):
                        next_token_logits[batch_idx, prev_token] /= repetition_penalty
            
            # Apply n-gram repetition penalty
            if no_repeat_ngram_size > 0:
                # For simplicity, we'll just check the last n-gram
                ngram_size = min(no_repeat_ngram_size, input_ids.size(1))
                for batch_idx in range(batch_size):
                    generated = input_ids[batch_idx, -ngram_size+1:].tolist() if ngram_size > 1 else []
                    banned_tokens = []
                    
                    # Find ngrams that would create a repetition
                    for prev_pos in range(input_ids.size(1) - ngram_size + 1):
                        prev_ngram = input_ids[batch_idx, prev_pos:prev_pos+ngram_size-1].tolist()
                        if prev_ngram == generated:
                            banned_token = input_ids[batch_idx, prev_pos+ngram_size-1].item()
                            banned_tokens.append(banned_token)
                    
                    # Apply mask to prevent repeated ngrams
                    for token in banned_tokens:
                        next_token_logits[batch_idx, token] = -float("inf")
            
            # Apply top-k filtering
            if top_k > 0:
                # Zero out all the values not in the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # If we encounter EOS tokens, avoid adding more tokens
            eos_token_id = config.get("eos_token_id", None)
            if eos_token_id is not None:
                next_token = next_token.masked_fill(
                    (input_ids == eos_token_id).any(dim=1).unsqueeze(1),
                    eos_token_id
                )
            
            # Append the sampled token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check if we've reached the end of generation
            if (input_ids == eos_token_id).any(dim=1).all().item():
                break
        
        return input_ids


class CrossModalFusionTransformer(nn.Module):
    """
    Transformer-based cross-modal fusion module.
    
    This module uses a stack of transformer layers to fuse information
    from vision and text modalities.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize the cross-modal fusion transformer.
        
        Args:
            dim: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Layer normalization for inputs
        self.vision_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the cross-modal fusion transformer.
        
        Args:
            vision_features: Vision features [batch_size, v_seq_len, dim]
            text_features: Text features [batch_size, t_seq_len, dim]
            text_attention_mask: Attention mask for text [batch_size, t_seq_len]
            
        Returns:
            Fused features [batch_size, v_seq_len + t_seq_len, dim]
        """
        # Apply layer normalization to inputs
        vision_features = self.vision_norm(vision_features)
        text_features = self.text_norm(text_features)
        
        # Concatenate features along sequence dimension
        combined_features = torch.cat([vision_features, text_features], dim=1)
        
        # Create attention mask for the combined sequence
        batch_size, v_seq_len, _ = vision_features.shape
        _, t_seq_len, _ = text_features.shape
        
        # Create vision attention mask (all True)
        vision_mask = torch.ones((batch_size, v_seq_len), dtype=torch.bool, device=vision_features.device)
        
        # Default text mask to all True if not provided
        if text_attention_mask is None:
            text_attention_mask = torch.ones((batch_size, t_seq_len), dtype=torch.bool, device=text_features.device)
        
        # Combine masks
        combined_mask = torch.cat([vision_mask, text_attention_mask], dim=1)
        
        # Process through transformer layers
        hidden_states = combined_features
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=combined_mask)
        
        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states


class CrossModalTransformerLayer(nn.Module):
    """
    A single transformer layer for cross-modal fusion.
    
    This layer includes self-attention, feed-forward network, and residual connections.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize a transformer layer.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Pre-normalization (for better training)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Transformed features [batch_size, seq_len, dim]
        """
        # Convert boolean mask to float attention mask
        if attention_mask is not None:
            # Create an additive attention mask (-inf for padding tokens)
            attn_mask = (1 - attention_mask.float()) * -10000.0
            # Expand mask dimensions to match what nn.MultiheadAttention expects
            expanded_mask = attn_mask.unsqueeze(1).expand(-1, x.size(1), -1)
        else:
            expanded_mask = None
        
        # Self-attention block (pre-norm architecture)
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=None if expanded_mask is None else ~attention_mask,
            need_weights=False
        )
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward block (pre-norm architecture)
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x


class GatedMultimodalFusion(nn.Module):
    """
    Gated fusion module for combining representations from different modalities.
    
    This module uses a gating mechanism to control the information flow between
    modalities, which helps balance their contributions.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize the gated fusion module.
        
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            output_dim: Dimension of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Projections for gating mechanism
        self.vision_gate = nn.Sequential(
            nn.Linear(vision_dim + text_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.text_gate = nn.Sequential(
            nn.Linear(vision_dim + text_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Projections for feature transformation
        self.vision_transform = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # Final fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the gated fusion module.
        
        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Concatenate features for gate computation
        concat_features = torch.cat([vision_features, text_features], dim=1)
        
        # Compute gates
        vision_gate_values = self.vision_gate(concat_features)
        text_gate_values = self.text_gate(concat_features)
        
        # Transform features
        vision_transformed = self.vision_transform(vision_features)
        text_transformed = self.text_transform(text_features)
        
        # Apply gating
        gated_vision = vision_transformed * vision_gate_values
        gated_text = text_transformed * text_gate_values
        
        # Combine gated features and apply final projection
        fused_features = gated_vision + gated_text
        fused_features = self.fusion_projection(fused_features)
        
        return fused_features


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    import os
    
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements a multimodal decoder generation model for text generation conditioned on multiple modalities",
        "key_classes": [
            {
                "name": "MultimodalDecoderGeneration",
                "purpose": "Generative model that combines vision and text inputs to generate text outputs",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vision_model: nn.Module, text_encoder: nn.Module, text_decoder: nn.Module, fusion_dim: int = 768, num_fusion_layers: int = 2, use_cross_attention: bool = True, max_sequence_length: int = 128, fusion_dropout: float = 0.1, tie_embeddings: bool = True, vocab_size: Optional[int] = None, use_gated_fusion: bool = True, generation_config: Optional[Dict[str, Any]] = None)",
                        "brief_description": "Initialize the multimodal decoder generation model"
                    },
                    {
                        "name": "prepare_inputs_for_decoder",
                        "signature": "prepare_inputs_for_decoder(self, vision_features: Optional[torch.Tensor] = None, encoder_features: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]",
                        "brief_description": "Prepare fused representation for the decoder"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, images: Optional[torch.Tensor] = None, encoder_input_ids: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.Tensor] = None, decoder_attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Process inputs through the model to compute logits and loss"
                    },
                    {
                        "name": "generate",
                        "signature": "generate(self, images: Optional[torch.Tensor] = None, encoder_input_ids: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, generation_config: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor",
                        "brief_description": "Generate text based on visual and/or textual inputs"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "typing"]
            },
            {
                "name": "CrossModalFusionTransformer",
                "purpose": "Transformer-based fusion mechanism for combining vision and text representations",
                "inheritance": "nn.Module"
            },
            {
                "name": "GatedMultimodalFusion",
                "purpose": "Gated fusion module that controls information flow between modalities",
                "inheritance": "nn.Module"
            }
        ],
        "external_dependencies": ["torch", "math", "logging"],
        "complexity_score": 9  # High complexity due to multiple components and generation functionality
    }