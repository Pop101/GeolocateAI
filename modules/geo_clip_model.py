import torch
import torch.nn as nn
import torch.optim as optim

from modules.clip_model import ClipBaseModel, ClipOutput
from modules.skipattnmlp import SkipAttentionMLP
from modules.feature_perspective import FeaturePerspective

class GeoClipModel:
    def __init__(self, lr=0.001, num_classes=150_000, num_hidden_dims = 1024*2, clip_model_name="openai/clip-vit-large-patch14", output_type=ClipOutput.POOLER_OUTPUT, device=None, dtype=torch.float32):   
        self.device = device
        self.dtype = dtype
        
        # Initialize model layers
        clip_model = ClipBaseModel(clip_model_name, output_type)
        
        # Create a single sequential model
        self.model = nn.Sequential(
            # Clip Embed (TODO: consider using vision transformer)
            clip_model,
            nn.LayerNorm(clip_model.logits_dim),
            
            # Project to hidden dimensions if needed
            nn.Linear(clip_model.logits_dim, num_hidden_dims) if clip_model.logits_dim != num_hidden_dims else nn.Identity(), 
            
            # Get feature perspective (different activation functions with attention)
            FeaturePerspective(num_hidden_dims, num_hidden_dims),
            nn.LayerNorm(num_hidden_dims),
            
            # Skip attention MLP (d=4) for classification
            SkipAttentionMLP(num_hidden_dims, out_features=num_classes),
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
            print(name) 
                
        self.optimizer = torch.optim.AdamW([
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
    
    def train_batch(self, batch, transforms=None):
        self.model.train()
        
        # Unpack batch
        images, logits = batch
        
        # Apply transforms on-the-fly to each image if transforms provided
        if transforms:
            transformed_images = transforms(images)
        else:
            transformed_images = images            
        
        # Move to device
        transformed_images = transformed_images.to(self.device)
        logits = logits.to(self.device)
                    
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(transformed_images)
        outputs = outputs.squeeze()
        loss = self.criterion(outputs, logits)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
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
                
                inputs = inputs.to(self.device)
                logits = logits.to(self.device)
            
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
    
    def send_to_device(self, device, dtype=None):
        """Sends the current model to the specified device and dtype, mutating the model (not like .to)"""
        if dtype is None:
            dtype = self.dtype
            
        self.model = self.model.to(device=device, dtype=dtype)
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