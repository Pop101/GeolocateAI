import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import List, Optional, Callable, Literal

def _compute_class_weights(labels: np.ndarray, weight_fn: Callable[[np.ndarray], np.ndarray]) -> torch.DoubleTensor:
    """Compute sample weights based on class frequencies and a weighting function."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = weight_fn(counts)
    
    # Map class weights to sample weights
    label_to_weight = dict(zip(unique_labels, class_weights))
    sample_weights = np.array([label_to_weight[label] for label in labels])
    
    # Normalize and convert to torch
    sample_weights = sample_weights / sample_weights.sum()
    return torch.DoubleTensor(sample_weights)

def _create_sampler(labels: List[int], weight_fn: Callable[[np.ndarray], np.ndarray], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Generic sampler creator with custom weighting function."""
    labels = np.array(labels)
    weights = _compute_class_weights(labels, weight_fn)
    num_samples = num_samples or len(labels)
    return WeightedRandomSampler(weights, num_samples, replacement=True)

def create_sqrt_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/sqrt(class_freq). Moderate rebalancing."""
    return _create_sampler(labels, lambda counts: 1.0 / np.sqrt(counts), num_samples)

def create_linear_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/class_freq. Full rebalancing (uniform across classes)."""
    return _create_sampler(labels, lambda counts: 1.0 / counts, num_samples)

def create_log_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/log(class_freq + 1). Gentle rebalancing."""
    return _create_sampler(labels, lambda counts: 1.0 / np.log1p(counts), num_samples)