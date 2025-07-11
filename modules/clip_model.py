import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from enum import Enum

class ClipOutput(Enum):
    """Enum for the output type of the CLIP model"""
    POOLER_OUTPUT = "pooler_output"
    LAST_HIDDEN_STATE = "last_hidden_state"

torch.serialization.add_safe_globals([ClipOutput])

class ClipBaseModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze=False, 
                 enable_checkpointing=False, output_type=ClipOutput.POOLER_OUTPUT):
        super().__init__()
        
        self.clip_vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.output_type = output_type
        self.enable_checkpointing = enable_checkpointing
        
        if enable_checkpointing:
            self.clip_vision_model.gradient_checkpointing_enable()
        
        for param in self.clip_vision_model.parameters():
            param.requires_grad = not freeze
    
    @torch.compiler.disable
    @torch.jit.ignore
    def _run_clip_forward(self, x):
        """Separate method to run CLIP without compilation"""
        return self.clip_vision_model(pixel_values=x)
    
    def forward(self, x):
        # Call the disabled method
        outputs = self._run_clip_forward(x)
        
        if self.output_type == ClipOutput.POOLER_OUTPUT:
            return outputs.pooler_output
        elif self.output_type == ClipOutput.LAST_HIDDEN_STATE:
            return outputs.last_hidden_state
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")
    
    @property
    def logits_dim(self):
        return self.clip_vision_model.config.hidden_size
    
    @property
    def is_frozen(self):
        return all(not param.requires_grad for param in self.clip_vision_model.parameters())