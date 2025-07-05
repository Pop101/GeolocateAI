import torch
import torch.nn as nn
import numpy as np

def mspace(start, end, num):
    factor = (end / start) ** (1 / num)
    return [start * (factor ** i) for i in range(num)]

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class SkipAttentionMLP(nn.Module):
    def __init__(self, in_features, out_features, depth=4, dropout_start=0.1, dropout_end=0.05):
        super().__init__()
        
        self.depth = depth
        
        # Dims calculation for each layer
        self.dims = mspace(in_features, max(out_features, 64), depth + 1) + [out_features]
        self.dims = list(map(round, self.dims))
        
        # Dropout values for each layer
        dropout_values = np.linspace(dropout_start, dropout_end, depth)
        dropout_values = [float(val) for val in dropout_values]
        
        # Create main processing blocks
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = nn.Sequential(
                nn.Linear(self.dims[i], self.dims[i + 1]),
                nn.LayerNorm(self.dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout_values[i])
            )
            self.blocks.append(block)
        
        # Create skip connections (from layer n to layer n+2)
        self.skip_connections = nn.ModuleList()
        for i in range(depth - 2):  # -2 because skip connections go two layers ahead
            skip = nn.Sequential(
                nn.Linear(self.dims[i + 1], self.dims[i + 3]),
                nn.LayerNorm(self.dims[i + 3])
            )
            self.skip_connections.append(skip)
        
        self.attention = nn.Sequential(
            nn.Linear(self.dims[-2], self.dims[-2]),
            nn.Sigmoid()
        )
        
        self.output = nn.Linear(self.dims[-2], out_features)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Store intermediate outputs for skip connections
        intermediates = []
        current = x
        
        # Forward through main blocks
        for i, block in enumerate(self.blocks):
            current = block(current)
            intermediates.append(current)
            
            # Apply skip connections if available (from 2 layers back)
            if i >= 2 and i - 2 < len(self.skip_connections):
                skip_connection = self.skip_connections[i - 2]
                # FIX: Use intermediates[i-2] instead of intermediates[i-1]
                current = current + skip_connection(intermediates[i-2])
        
        # Apply attention mechanism
        att = self.attention(current)
        current = current * att
        
        return self.output(current)