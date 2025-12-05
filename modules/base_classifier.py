import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import os
from collections import defaultdict

class AbstractGeoClassifier(ABC):
    """
    Abstract base class for geolocation classifiers.
    Manages common lifecycle: optimizer, scheduler, device transfer, save/load.
    """
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.total_batches_trained = 0
        self.init_params = {}

    @abstractmethod
    def train_batch(self, batch, transforms=None) -> float:
        """Train on a single batch. Returns loss."""
        pass

    @abstractmethod
    def evaluate(self, data_loader, transforms=None) -> Tuple[float, float]:
        """Evaluate on a dataset. Returns (loss, accuracy)."""
        pass

    @abstractmethod
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict logits for an image."""
        pass

    def update_scheduler(self, val_loss):
        if self.scheduler:
            self.scheduler.step(val_loss)

    def get_current_lr(self):
        if self.optimizer:
            return max(group['lr'] for group in self.optimizer.param_groups)
        return 0.0

    def send_to_device(self, device, dtype=None):
        """Moves the model and all its components to the specified device."""
        if dtype is None:
            dtype = self.dtype
        
        self.device = device
        self.dtype = dtype

        if self.criterion:
            self.criterion.to(device)
            
        # Move optimizer state
        if self.optimizer:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if v.is_floating_point():
                            state[k] = v.to(device=device, dtype=dtype)
                        else:
                            state[k] = v.to(device=device)

        # Subclasses should override to move their specific components, then call super()
        # or handle base movement themselves.

    def save(self, filepath: str):
        """Standardized save method."""
        state = {
            'init_params': self.init_params,
            'total_batches_trained': self.total_batches_trained,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        state.update(self._get_additional_save_state())
        torch.save(state, filepath)

    def _get_additional_save_state(self) -> Dict[str, Any]:
        """Subclasses can return additional state to save."""
        return {}

    @classmethod
    def load(cls, filepath: str, map_location=None):
        """Standardized load method."""
        if map_location is None:
            map_location = torch.device('cpu')
            
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Reconstruct model
        model = cls(**checkpoint['init_params'])
        
        model.total_batches_trained = checkpoint.get('total_batches_trained', 0)
        
        # Load subclass specific state FIRST (may create param groups needed for optimizer)
        model._load_additional_state(checkpoint)
        
        # Load optimizer/scheduler AFTER subclass state (param groups must match)
        if model.optimizer and checkpoint.get('optimizer_state_dict'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if model.scheduler and checkpoint.get('scheduler_state_dict'):
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model

    def _load_additional_state(self, checkpoint: Dict[str, Any]):
        """Subclasses can load additional state."""
        pass

    def get_model_size(self):
        """Returns the ON-DEVICE size of the model in bytes."""
        return 0
