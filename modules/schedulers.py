import torch
import numpy as np

class SmoothReduceLROnPlateau:
    """
    A learning rate scheduler that reduces the learning rate when a plateau in loss is detected,
    based on smoothed recent loss compared to smoothed historical loss.
    
    If the recent average loss exceeds the historical average loss by a certain threshold,
    the learning rate is reduced by a specified factor.
    """
    def __init__(self, optimizer, 
                 smoothing_window=10, 
                 historical_window=100,
                 reduction_threshold=0.95,
                 cooldown=20,
                 factor=0.8,
                 min_lr=1e-6,
                 verbose=False):
        
        self.optimizer = optimizer
        self.smoothing_window = smoothing_window
        self.historical_window = historical_window
        self.reduction_threshold = reduction_threshold
        self.cooldown = cooldown
        self.factor = factor
        self.min_lr = min_lr
        self.losses = []
        self.verbose = verbose
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def step(self, metrics):
        self.losses.append(metrics)
        while len(self.losses) > self.historical_window:
            self.losses.pop(0)
        if len(self.losses) < self.historical_window:
            return
        
        recent_avg = np.mean(self.losses[-self.smoothing_window:])
        historical_avg = np.mean(self.losses[:-self.smoothing_window])
        
        if recent_avg > historical_avg * self.reduction_threshold:
            if self.verbose:
                print(f'Reducing learning rate: recent_avg={recent_avg:.4f} > '
                      f'historical_avg={historical_avg:.4f} * {self.reduction_threshold}')
                
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = max(group['lr'] * self.factor, self.min_lr)
                if self.verbose: print(f'Reducing learning rate of group {i} to {group['lr']:.4e}.')
        
            if self.cooldown > 0:
                self.losses = self.losses[:-self.cooldown]
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]