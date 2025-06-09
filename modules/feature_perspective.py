import torch
import torch.nn as nn

class FeaturePerspective(nn.Module):
    """
    Classification head that applies different activation functions with attention
    """
    def __init__(self, input_dim, output_dim,
                 num_heads=16, dropout=0.10, 
                 activation_perspectives=[nn.GELU(), nn.SiLU(), nn.Tanh()]
                 ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Apply different activation functions to the features
        self.feature_extractors = nn.ModuleList()
        for activation in activation_perspectives:
            self.feature_extractors.append(nn.Sequential(
                nn.Linear(input_dim, input_dim),
                activation,
                nn.Dropout(dropout)
            ))
        
        # Multi-head attention for feature interaction
        self.attention = nn.Sequential(
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout)
        )
        
        # Project features to output_dim before cross-attention
        self.project_features = nn.Linear(input_dim, output_dim)
        
        # Cross-attention with learnable geographic queries
        self.geo_queries = nn.Parameter(torch.randn(8, output_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.refinement = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for extractor in self.feature_extractors:
            nn.init.kaiming_normal_(extractor[0].weight, nonlinearity='relu')
            nn.init.constant_(extractor[0].bias, 0)
        
        nn.init.kaiming_normal_(self.project_features.weight, nonlinearity='relu')
    
    def forward(self, x):
        """
        Forward pass through the classification head.
        
        Args:
            x: CLIP features of shape (batch_size, input_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        # Apply different activation functions to the features
        features = [extractor(x) for extractor in self.feature_extractors]
        
        # Multi-head attention for feature interaction
        features = self.attention(torch.cat(features, dim=1))
        
        # Project to output_dim
        features = self.project_features(features)
        
        # Cross-attention with learnable geographic queries
        geo_queries = self.geo_queries.unsqueeze(0).repeat(x.size(0), 1, 1)
        features = self.cross_attention(features, geo_queries, features)
        
        # Final refinement
        return self.refinement(features)
