import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPVisionModel

from enum import Enum


class ClipOutput(Enum):
    """Enum for the output type of the CLIP model"""
    POOLER_OUTPUT = "pooler_output"
    LAST_HIDDEN_STATE = "last_hidden_state"
torch.serialization.safe_globals([ClipOutput])    
    
class ClipBaseModel(nn.Module):
    """Wrapper for CLIP vision model to use in a sequential model"""
    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze=False, enable_checkpointing=False, output_type=ClipOutput.POOLER_OUTPUT):
        super().__init__()
        self.clip_vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.output_type = output_type
        self.enable_checkpointing = enable_checkpointing
        
        # Freeze the model
        for param in self.clip_vision_model.parameters():
            param.requires_grad = not freeze
        
    def forward(self, x):
        if self.enable_checkpointing:
            if self.training: # use gradient checkpointing
                self.clip_vision_model.gradient_checkpointing_enable()
            else: # disable gradient checkpointing for inference
                self.clip_vision_model.gradient_checkpointing_disable()
                
        outputs = self.clip_vision_model(pixel_values=x)
        if self.output_type == ClipOutput.POOLER_OUTPUT:
            return outputs.pooler_output
        elif self.output_type == ClipOutput.LAST_HIDDEN_STATE:
            return outputs.last_hidden_state
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")
    
    @property
    def logits_dim(self):
        """Returns the dimensions of the model's output logits"""
        return self.clip_vision_model.config.hidden_size
    
    @property
    def is_frozen(self):
        return all(not param.requires_grad for param in self.clip_vision_model.parameters())