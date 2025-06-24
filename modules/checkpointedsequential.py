import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedSequential(nn.Module):
    """A Sequential-like module that applies gradient checkpointing to specified layers"""
    
    def __init__(self, *layers, checkpoint_segments=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
        if checkpoint_segments is None:
            checkpoint_segments = []
            for i in range(len(self.layers)):
                # Default to checkpoint if num params in layer is large (> 1M)
                if sum(p.numel() for p in self.layers[i].parameters()) > 1_000_000:
                    checkpoint_segments.append(i)
        
        elif isinstance(checkpoint_segments, int):
            # If a single int is provided, checkpoint just that segment
            checkpoint_segments = [checkpoint_segments]
        
        elif not hasattr(checkpoint_segments, '__iter__'):
            raise TypeError("checkpoint_segments must be None, an int, or an iterable of ints")
        
        self.checkpoint_segments = checkpoint_segments or []
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.training and i in self.checkpoint_segments:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
    
    