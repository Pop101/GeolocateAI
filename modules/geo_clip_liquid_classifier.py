import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb

from modules.clip_model import ClipBaseModel, ClipOutput
from modules.skipattnmlp import SkipAttentionMLP
from modules.feature_perspective import FeaturePerspective
from modules.checkpointedsequential import CheckpointedSequential
from modules.kldivlosssoftmax import KLDivLossWithSoftmax
from modules.schedulers import SmoothReduceLROnPlateau

import warnings

class GeoLiquidClipModel:
    def __init__(self, lr=0.001, num_classes=150_000, num_head_dims=1024*2, num_hidden_dims=1024*8, heads=32, depth=8, enable_checkpointing=False, clip_model_name="openai/clip-vit-large-patch14", output_type=ClipOutput.POOLER_OUTPUT, device=None, dtype=torch.float32):   
        self.device = device
        self.dtype = dtype
        
        # Initialize CLIP base model separately
        self.base = ClipBaseModel(clip_model_name, output_type=output_type, enable_checkpointing=enable_checkpointing)
        
        # Create the classifier head model
        seq = CheckpointedSequential if enable_checkpointing else nn.Sequential
        self.model = seq(
            # LayerNorm after CLIP
            nn.LayerNorm(self.base.logits_dim),
            
            # Project to hidden dimensions if needed
            nn.Linear(self.base.logits_dim, num_head_dims) if self.base.logits_dim != num_head_dims else nn.Identity(), 
            
            # Get feature perspective (different activation functions with attention)
            FeaturePerspective(num_head_dims, num_head_dims, num_heads=heads),
            
            # Use skipattn to draw from the feature perspective
            SkipAttentionMLP(num_head_dims, num_hidden_dims, depth=depth),
            
            # Single linear expand to logits
            nn.Linear(num_hidden_dims, num_classes)
        )
        
        # Initialize criterion and optimizer
        self.criterion = KLDivLossWithSoftmax()
        
        # Optimizer with different learning rates for different components
        clip_params = list(self.base.parameters())
        geo_processor_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'feature_perspective' in name.lower():
                geo_processor_params.append(param)
            else:
                classifier_params.append(param)

        self.optimizer = bnb.optim.AdamW8bit([
            {'params': clip_params, 'lr': lr * 0.05},         # Lowest LR for pretrained CLIP
            {'params': geo_processor_params, 'lr': lr * 0.5}, # Medium LR for geo reasoning
            {'params': classifier_params, 'lr': lr}           # Highest LR for final classifier
        ], lr=lr, weight_decay=1e-4)
        
        self.scheduler = SmoothReduceLROnPlateau(
            self.optimizer,
            smoothing_window=10,
            historical_window=100,
            reduction_threshold=0.95,
            cooldown=20,
            factor=0.8,
            min_lr=1e-5,
        )
        
        # Use device parameter if provided
        if device is not None or dtype is not None:
            self.send_to_device(device, dtype)
            
        # Save init params for future use
        self.total_batches_trained = 0
        self.init_params = {
            'lr': lr,
            'num_classes': num_classes,
            'num_head_dims': num_head_dims,
            'num_hidden_dims': num_hidden_dims,
            'heads': heads,
            'depth': depth,
            'clip_model_name': clip_model_name,
            'output_type': output_type,
        }        
    
    def train_batch(self, batch, transforms=None, accumulation_steps=3):
        self.base.train()
        self.model.train()
        
        # Unpack Batch
        images, logits = batch
        
        # Apply transforms on-the-fly
        if transforms:
            images = transforms(images)
        
        # Split batch for accumulation
        batch_size = images.size(0) // accumulation_steps
        
        accumulated_loss = 0
        for i in range(accumulation_steps):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            sub_images = images[start_idx:end_idx]
            sub_logits = logits[start_idx:end_idx]
            
            # Forward pass through base and model
            clip_features = self.base(sub_images)
            outputs = self.model(clip_features)
            loss = self.criterion(outputs, sub_logits) / accumulation_steps
            if loss.isnan().any():
                warnings.warn(f"NaN loss encountered in batch {i+1}/{accumulation_steps}. Skipping this batch.")
                continue
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
        
        # Update weights after accumulation
        torch.nn.utils.clip_grad_norm_(list(self.base.parameters()) + list(self.model.parameters()), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.total_batches_trained += 1
        return accumulated_loss
    
    def update_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
    
    def evaluate(self, data_loader, transforms=None):
        """Evaluate the model on the validation set
        Returns:
            average loss: float
            average hit rate: float
        """
        
        self.base.eval()
        self.model.eval()
        total_loss = 0.0
        total_hits = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, logits in data_loader:                
                if transforms:
                    inputs = transforms(inputs)
                
                inputs = inputs.to(self.device, dtype=self.dtype)
                logits = logits.to(self.device, dtype=self.dtype)
            
                # Get output floats
                clip_features = self.base(inputs)
                outputs = self.model(clip_features)
                outputs = outputs.view(logits.shape)
                loss = self.criterion(outputs, logits)
                
                if loss.isnan().any():
                    warnings.warn("NaN loss encountered during evaluation. Skipping this batch.")
                    continue
                
                # Get probabilities from logits
                input_prob  = torch.softmax(logits, dim=-1)
                output_prob = torch.softmax(outputs, dim=-1)
                
                # Tallies
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size # loss retrurns element-averaged, we want total
                total_hits += (output_prob.argmax(dim=1) == input_prob.argmax(dim=1)).float().sum()
                num_samples += batch_size
        
        return total_loss / num_samples, total_hits / num_samples
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted logits for a single image"""
        self.base.eval()
        self.model.eval()
        with torch.no_grad():
            clip_features = self.base(image)
            return self.model(clip_features)
        
    def predict_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted probabilities for a single image"""
        logits = self(image)
        return torch.softmax(logits, dim=-1)
        
    def get_current_lr(self):
        """Return the current learning rate"""
        return max(group['lr'] for group in self.optimizer.param_groups)
    
    def get_model_size(self):
        """Returns the ON-DEVICE size of the model in bytes"""
        total_size = 0
        for param in list(self.base.parameters()) + list(self.model.parameters()):
            total_size += param.numel() * param.element_size()
        
        for buffer in list(self.base.buffers()) + list(self.model.buffers()):
            total_size += buffer.numel() * buffer.element_size()
        
        return total_size
    
    def send_to_device(self, device, dtype=None):
        """Sends the current model to the specified device and dtype, mutating the model"""
        if dtype is None:
            dtype = self.dtype
        
        # Move base model
        for param in self.base.parameters():
            param.data = param.data.to(device=device, dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device=device, dtype=dtype)
        
        for buffer in self.base.buffers():
            if buffer.dtype in [torch.int32, torch.int64, torch.long]:
                buffer.data = buffer.data.to(device=device)
            else:
                buffer.data = buffer.data.to(device=device, dtype=dtype)
        
        # Move classifier model
        for param in self.model.parameters():
            param.data = param.data.to(device=device, dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device=device, dtype=dtype)
        
        for buffer in self.model.buffers():
            if buffer.dtype in [torch.int32, torch.int64, torch.long]:
                buffer.data = buffer.data.to(device=device)
            else:
                buffer.data = buffer.data.to(device=device, dtype=dtype)
        
        self.criterion = self.criterion.to(device)
        self.device = device
        self.dtype = dtype
        
    def compile(self, **kwargs):
        """Compiles the model for optimized inference."""
        # Only compile the classifier head, not the CLIP base
        self.model = torch.compile(self.model, **kwargs)
    
    def save(self, filepath):
        # Extract state dict from compiled or regular model
        if hasattr(self.model, '_orig_mod'):
            model_state_dict = self.model._orig_mod.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        torch.save({
            'base_state_dict': self.base.state_dict(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'init_params': self.init_params,
            'total_batches_trained': self.total_batches_trained
        }, filepath)
    
    @staticmethod
    def load(filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model with saved init params
        model = GeoLiquidClipModel(**checkpoint['init_params'])
        
        # Load states
        model.base.load_state_dict(checkpoint['base_state_dict'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training progress
        model.total_batches_trained = checkpoint['total_batches_trained']
        
        return model