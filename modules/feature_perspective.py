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
        
        # Round up to nearest multiple of num_heads
        def round_up_to_multiple(x, multiple):
            return ((x + multiple - 1) // multiple) * multiple
        
        nearest_valid_input_dim = round_up_to_multiple(input_dim, num_heads)
        
        # Apply different activation functions to the features
        self.feature_extractors = nn.ModuleList()
        for activation in activation_perspectives:
            self.feature_extractors.append(nn.Sequential(
                nn.Linear(input_dim, nearest_valid_input_dim),
                activation,
                nn.Dropout(dropout)
            ))
        
        # Multi-head attention for feature interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=nearest_valid_input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(nearest_valid_input_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Project features to output_dim before cross-attention
        nearest_valid_output_dim = round_up_to_multiple(output_dim, num_heads)
        self.project_features = nn.Linear(nearest_valid_input_dim, nearest_valid_output_dim)
        
        # Cross-attention with learnable geographic queries
        self.geo_queries = nn.Parameter(torch.randn(8, nearest_valid_output_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=nearest_valid_output_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.refinement = nn.Sequential(
            nn.Linear(nearest_valid_output_dim, output_dim),
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
        
        # Stack features for attention (batch_size, num_perspectives, input_dim)
        features_stacked = torch.stack(features, dim=1)
        
        # Multi-head self-attention for feature interaction
        attn_output, _ = self.attention(features_stacked, features_stacked, features_stacked)
        attn_output = self.attention_norm(attn_output)
        attn_output = self.attention_dropout(attn_output)
        
        # Average across perspectives or take the first one
        features = attn_output.mean(dim=1)  # (batch_size, input_dim)
        
        # Project to output_dim
        features = self.project_features(features)
        features = features.unsqueeze(1)  # Add sequence dimension for cross-attention
        
        # Cross-attention with learnable geographic queries
        geo_queries = self.geo_queries.unsqueeze(0).repeat(x.size(0), 1, 1)
        features, _ = self.cross_attention(geo_queries, features, features)
        
        # Final refinement - take mean across geographic queries
        features = features.mean(dim=1)  # (batch_size, output_dim)
        return self.refinement(features)