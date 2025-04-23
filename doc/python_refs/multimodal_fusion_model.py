import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    """
    End-to-end model for multimodal fusion of text and images.
    Combines text and image processing with cross-modal attention
    for integrated understanding.
    """
    def __init__(self, 
                 vocab_size, 
                 embed_dim=512, 
                 text_encoder_layers=6,
                 text_encoder_heads=8,
                 image_size=224,
                 patch_size=16):
        super().__init__()
        
        # Text encoder components
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_position_embedding = nn.Parameter(
            torch.zeros(1, 512, embed_dim)  # 512 is max sequence length
        )
        
        # Text encoder
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=text_encoder_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.text_encoder = nn.TransformerEncoder(
            text_encoder_layer, 
            num_layers=text_encoder_layers
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(  # Assumes ImageEncoder is imported
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Cross-modal fusion
        self.text_to_image_attention = CrossModalAttention(  # Assumes CrossModalAttention is imported
            embed_dim=embed_dim,
            num_heads=text_encoder_heads
        )
        
        self.image_to_text_attention = CrossModalAttention(  # Assumes CrossModalAttention is imported
            embed_dim=embed_dim,
            num_heads=text_encoder_heads
        )
        
        # Final projection for output (e.g., for classification or generation)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the model"""
        nn.init.normal_(self.text_position_embedding, std=0.02)
    
    def forward(self, text_ids, images):
        """
        Forward pass for the multimodal model
        
        Args:
            text_ids: Token IDs [batch_size, seq_len]
            images: Image tensors [batch_size, channels, height, width]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        # Process text
        text_embeddings = self.token_embedding(text_ids)
        # Add position embeddings for text
        seq_length = text_embeddings.size(1)
        text_embeddings = text_embeddings + self.text_position_embedding[:, :seq_length, :]
        
        # Pass through text encoder
        # Need to convert from [batch, seq, dim] to [seq, batch, dim] for nn.TransformerEncoder
        text_features = self.text_encoder(text_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Process images
        image_features = self.image_encoder(images)
        
        # Apply bidirectional cross-modal attention
        text_with_image_context = self.text_to_image_attention(text_features, image_features)
        image_with_text_context = self.image_to_text_attention(image_features, text_features)
        
        # Simple concatenation of the first token from each modality for final representation
        # More sophisticated fusion could be used here
        multimodal_representation = text_with_image_context
        
        # Generate output logits
        output = self.output_projection(multimodal_representation)
        
        return output