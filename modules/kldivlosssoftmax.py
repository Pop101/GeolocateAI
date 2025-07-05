import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLossWithSoftmax(nn.Module):
    """A loss function useful for training models"""
    def __init__(self, reduction='batchmean'):
        super(KLDivLossWithSoftmax, self).__init__()
        self.reduction = reduction
    
    def forward(self, input_logits, target_distributions):
        """
        Args:
            input_logits: Raw logits from model (batch_size, num_classes)
            target_distributions: Target probability distributions (batch_size, num_classes)
        
        Returns:
            KL divergence loss
        """
        
        # Apply log_softmax to input logits
        log_probs = F.log_softmax(input_logits, dim=1)
        
        # Compute KL divergence
        return F.kl_div(log_probs, target_distributions, reduction=self.reduction)