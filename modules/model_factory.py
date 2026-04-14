import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from modules.clip_model import ClipBaseModel, ClipOutput
from modules.feature_perspective import FeaturePerspective
from modules.skipattnmlp import SkipAttentionMLP

class ModelFactory(ABC):
    """
    Abstract factory for creating model components.
    """
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def create_model(self, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def get_optimizer_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        pass

class ClipBackboneFactory(ModelFactory):
    """
    Factory for creating the CLIP backbone model.
    """
    def __init__(self, 
                 clip_model_name="openai/clip-vit-large-patch14", 
                 output_type=ClipOutput.POOLER_OUTPUT,
                 enable_checkpointing=False):
        self.clip_model_name = clip_model_name
        self.output_type = output_type
        self.enable_checkpointing = enable_checkpointing
        self._output_dim = 0

    @property
    def input_dim(self) -> int:
        return 0 # Image input

    @property
    def output_dim(self) -> int:
        if self._output_dim == 0:
             # Heuristic based on name if not yet created
             # ViT-L/14 hidden_size is 1024, ViT-B/32 is 768
             if "large" in self.clip_model_name: self._output_dim = 1024 
             elif "base" in self.clip_model_name: self._output_dim = 768
             else: self._output_dim = 768 
        return self._output_dim

    def create_model(self, freeze=False, **kwargs) -> nn.Module:
        model = ClipBaseModel(
            self.clip_model_name, 
            output_type=self.output_type, 
            freeze=freeze, 
            enable_checkpointing=self.enable_checkpointing
        )
        self._output_dim = model.logits_dim 
        return model

    def get_optimizer_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        return [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr * 0.1}]

class SkipAttentionMLPFactory(ModelFactory):
    """
    Factory for creating the SkipAttentionMLP classifier head.
    """
    def __init__(self, input_dim: int, output_dim: int = 0, num_hidden_dims=2048, heads=8, depth=5):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.num_hidden_dims = num_hidden_dims
        self.heads = heads
        self.depth = depth

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def create_model(self, output_dim: int = None, **kwargs) -> nn.Module:
        # Allow overriding output_dim at creation time (useful for hierarchical levels)
        final_output_dim = output_dim if output_dim is not None else self.output_dim
        
        assert self.input_dim > 0, f"input_dim must be positive, got {self.input_dim}"
        assert final_output_dim > 0, f"output_dim must be positive, got {final_output_dim}"
        assert self.num_hidden_dims > 0, f"num_hidden_dims must be positive, got {self.num_hidden_dims}"
        
        return nn.Sequential(
            FeaturePerspective(self.input_dim, self.input_dim, num_heads=self.heads),
            SkipAttentionMLP(
                in_features=self.input_dim,
                out_features=self.num_hidden_dims,
                depth=self.depth
            ),
            nn.Linear(self.num_hidden_dims, final_output_dim)
        )

    def get_optimizer_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        return [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr}]
