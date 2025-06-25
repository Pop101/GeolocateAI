import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb

from modules.clip_model import ClipBaseModel, ClipOutput
from modules.skipattnmlp import SkipAttentionMLP
from modules.feature_perspective import FeaturePerspective
from modules.checkpointedsequential import CheckpointedSequential

class GeoClipModel:
    def __init__(self, lr=0.001, num_classes=150_000, num_hidden_dims = 1024, clip_model_name="openai/clip-vit-large-patch14", output_type=ClipOutput.POOLER_OUTPUT, device=None, dtype=torch.float32):   
        self.device = device
        self.dtype = dtype
        
        # Initialize model layers
        clip_model = ClipBaseModel(clip_model_name, output_type)
        
        # Create a single sequential model
        self.model = CheckpointedSequential(
            # Clip Embed
            clip_model,
            nn.LayerNorm(clip_model.logits_dim),
            
            # Project to hidden dimensions if needed
            nn.Linear(clip_model.logits_dim, num_hidden_dims) if clip_model.logits_dim != num_hidden_dims else nn.Identity(), 
            
            # Get feature perspective (different activation functions with attention)
            FeaturePerspective(num_hidden_dims, num_hidden_dims, num_heads=16),
            
            # Single linear (no skip attn, too large for gpu)
            nn.Linear(num_hidden_dims, num_classes)
        )
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        
        # Optimizer with different learning rates for different components
        clip_params = []
        geo_processor_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'clip' in name.lower():
                clip_params.append(param)
            elif 'feature_perspective' in name.lower():
                geo_processor_params.append(param)
            else:
                classifier_params.append(param)

        self.optimizer = bnb.optim.AdamW8bit([
            {'params': clip_params, 'lr': lr * 0.05},         # Lowest LR for pretrained CLIP
            {'params': geo_processor_params, 'lr': lr * 0.5}, # Medium LR for geo reasoning
            {'params': classifier_params, 'lr': lr}           # Highest LR for final classifier
        ], lr=lr, weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            cooldown=3,
            min_lr=1e-6
        )
        
        # Use device parameter if provided
        if device is not None or dtype is not None:
            self.send_to_device(device, dtype)
    
    def train_batch(self, batch, transforms=None, accumulation_steps=3):
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
            
            # Forward pass
            outputs = self.model(sub_images)
            loss = self.criterion(outputs, sub_logits) / accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
        
        # Update weights after accumulation
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return accumulated_loss
    
    def update_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
    
    def evaluate(self, data_loader, transforms=None):
        """Evaluate the model on the validation set
        Returns:
            average loss: float
            average hit rate: float
        """
        
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
                outputs = self.model(inputs)
                outputs = outputs.view(logits.shape)
                loss = self.criterion(outputs, logits)
                
                # Tallies
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_hits += (outputs.argmax(dim=1) == logits.argmax(dim=1)).float().sum()
                num_samples += batch_size
        
        return total_loss / num_samples, total_hits / num_samples
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted probabilities for a single image"""
        self.model.eval()
        with torch.no_grad():
            return self.model(image)
        
    def get_current_lr(self):
        """Return the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_model_size(self):
        """Returns the ON-DEVICE size of the model in bytes"""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.numel() * param.element_size()
        
        for buffer in self.model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        return total_size
    
    def send_to_device(self, device, dtype=None):
        """Sends the current model to the specified device and dtype, mutating the model"""
        if dtype is None:
            dtype = self.dtype
        
        # Move parameters (these can change dtype)
        for param in self.model.parameters():
            param.data = param.data.to(device=device, dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device=device, dtype=dtype)
        
        # Move buffers (preserve integer types)
        for buffer in self.model.buffers():
            if buffer.dtype in [torch.int32, torch.int64, torch.long]:
                # Keep integer buffers as integers, just move device
                buffer.data = buffer.data.to(device=device)
            else:
                # Float buffers can change dtype
                buffer.data = buffer.data.to(device=device, dtype=dtype)
        
        self.criterion = self.criterion.to(device)
        self.device = device
        self.dtype = dtype
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
    
    @staticmethod
    def load(filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = GeoClipModel(lr=float("inf")) # lr will be overwritten by the loaded value
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model