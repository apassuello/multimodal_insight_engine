import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) style image encoder for multimodal integration.
    Processes images into a sequence of patch embeddings compatible with
    transformer architecture.
    """
    def __init__(self, 
                image_size=224, 
                patch_size=16, 
                in_channels=3,
                embed_dim=768):
        super().__init__()
        
        # Calculate parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Create patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Class token (similar to BERT's [CLS] token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the image encoder"""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Forward pass for the image encoder
        
        Args:
            x: Image tensor [batch_size, channels, height, width]
            
        Returns:
            Patch embeddings [batch_size, num_patches+1, embed_dim]
        """
        # x shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Create patch embeddings: [batch_size, num_patches, embed_dim]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x