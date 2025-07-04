import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, vit_l_16, ViT_L_16_Weights, vit_l_32, ViT_L_32_Weights

from enum import Enum


class VisionTransformerBase(Enum):
    """Enum for the base Vision Transformer model"""
    VIT_B_16 = vit_b_16, ViT_B_16_Weights.DEFAULT
    VIT_B_32 = vit_b_32, ViT_B_32_Weights.DEFAULT
    VIT_L_16 = vit_l_16, ViT_L_16_Weights.DEFAULT
    VIT_L_32 = vit_l_32, ViT_L_32_Weights.DEFAULT

torch.serialization.add_safe_globals([VisionTransformerBase])
    
class VisionTransformerModel(nn.Module):
    """Vision Transformer base model for image classification"""
    def __init__(self, input_size=224, base_model:VisionTransformerBase=VisionTransformerBase.VIT_B_32):
        super().__init__()
        
        # Initialize Vision Transformer     
        self.model = base_model[0](base_model[1])
        
        # Replace embedding layer with larger size
        if input_size != 224:
            # TODO: make this actually work
             self._modify_model_for_input_size(input_size)
        
        # Excise the classifier head
        self.model.heads = nn.Identity()
        
    def forward(self, x):
        """Forward pass of the Vision Transformer model"""
        return self.model(x)
    
    @property
    def logits_dim(self):
        """Returns the dimensions of the model's output logits"""
        return self.model.heads.in_features
    
    def _modify_model_for_input_size(self, input_size):
        """Modify the ViT model to accept a different input size"""
        # Update the image_size attribute to avoid assertion errors
        self.model.image_size = input_size
        
        # Save original positional embedding
        orig_pos_embed = self.model.encoder.pos_embedding
        
        # Original encoding size and patch size
        patch_size = self.model.patch_size
        orig_size = 224 // patch_size 
        new_size = input_size // patch_size
        
        # Extract class token and position embeddings
        cls_pos_embed = orig_pos_embed[:, 0:1]
        pos_embed = orig_pos_embed[:, 1:]
        
        # Reshape position embeddings to grid
        dim = pos_embed.shape[-1]
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        
        # Interpolate position embeddings to new size
        pos_embed = F.interpolate(
            pos_embed, 
            size=(new_size, new_size), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        # Concat with class token and update model
        new_pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        self.model.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        
        # Update the expected sequence length
        self.model.seq_length = new_pos_embed.shape[1]