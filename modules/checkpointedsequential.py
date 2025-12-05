import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

class CheckpointedSequential(nn.Module):
    """A Sequential-like module that applies gradient checkpointing to specified layers"""
    
    def __init__(self, *layers, checkpoint_segments=None):
        super().__init__()
        
        if checkpoint_segments is None:
            checkpoint_segments = []
            for i in range(len(layers)):
                # Default to checkpoint if num params in layer is large (> 1M)
                if sum(p.numel() for p in layers[i].parameters()) > 1_000_000:
                    checkpoint_segments.append(i)
        
        elif isinstance(checkpoint_segments, int):
            # If a single int is provided, checkpoint just that segment
            checkpoint_segments = [checkpoint_segments]
        
        elif not hasattr(checkpoint_segments, '__iter__'):
            raise TypeError("checkpoint_segments must be None, an int, or an iterable of ints")
        
        self.checkpoint_segments = checkpoint_segments or []
        
        # Use numbered attributes like nn.Sequential for state_dict compatibility
        for i, layer in enumerate(layers):
            if i in self.checkpoint_segments:
                checkpointed_layer = checkpoint_wrapper(
                    layer,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
                self.add_module(str(i), checkpointed_layer)
            else:
                self.add_module(str(i), layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x
    
    def get_checkpointed_indices(self):
        """Helper method to see which layer indices are checkpointed"""
        return self.checkpoint_segments.copy()
    
    def is_layer_checkpointed(self, index):
        """Check if a specific layer index is checkpointed"""
        return index in self.checkpoint_segments