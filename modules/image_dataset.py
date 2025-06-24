from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

from PIL import Image

from collections import OrderedDict

from typing import Tuple
import random

MEAN = [0.4907244145870209, 0.4381392002105713, 0.407711386680603]
STD  = [0.2914842963218689, 0.2807084321975708, 0.2719435691833496]

def pad_to_square(image:torch.Tensor) -> torch.Tensor:
    """Pad an image to make it square."""
    c, h, w = image.shape
    
    # Find the maximum dimension
    max_dim = max(h, w)
    
    # Calculate padding for height and width
    pad_h = (max_dim - h) // 2
    pad_h_remainder = (max_dim - h) % 2  # Handle odd padding
    
    pad_w = (max_dim - w) // 2
    pad_w_remainder = (max_dim - w) % 2  # Handle odd padding
    
    # Apply padding [left, right, top, bottom]
    padding = (pad_w, pad_w + pad_w_remainder, pad_h, pad_h + pad_h_remainder)
    
    # Use torch functional padding with constant value (usually 0 or 1 depending on normalization)
    padded_image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
    return padded_image
        
class ImageDataset(Dataset):
    def __init__(self, image_paths:list, output_values:list, size:tuple=(224, 224)):
        self.image_paths = image_paths
        self.output_values = output_values
        self.size = size
        
        assert len(image_paths) == len(output_values), "Length of image_paths and output_values must be the same"
    
    @staticmethod
    def get_transforms(train=True) -> transforms.Compose:
        """Create image transforms with learned mean and std"""        
        # Create transforms pipeline
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        else:
            return transforms.Compose([
                transforms.Normalize(mean=MEAN, std=STD),
            ])
            
    def get_size_transpose(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(pad_to_square),
            transforms.Resize(self.size),
        ])
        
    def __len__(self):
        return min(len(self.image_paths), len(self.output_values))
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        output_val = self.output_values[idx]
        
        # Load image. Note it is the user's job to apply transforms if desired
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.get_size_transpose()(image)
        return image_tensor, output_val


# If we ever need to recalculate stats:
def calculate_stats(data:ImageDataset, sample_size: int) -> Tuple[list, list]:
    """Calculate mean and std from a random sample of images"""
    # Take a sample of images
    sample_indices = random.sample(range(len(data)), min(sample_size, len(data)))
    
    # Convert to tensor (N, C, H, W)
    sample_tensors = []
    to_tensor = transforms.ToTensor()
    
    for idx in sample_indices:
        img_path = data.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(data.size)
        img_tensor = to_tensor(img)
        img_tensor = pad_to_square(img_tensor)
        sample_tensors.append(img_tensor)
    
    # Stack tensors and calculate stats
    if sample_tensors:
        sample_batch = torch.stack(sample_tensors)
        mean = sample_batch.mean(dim=[0, 2, 3]).tolist()
        std = sample_batch.std(dim=[0, 2, 3]).tolist()
        return mean, std
    
    raise ValueError("No images found in the sample.")

if __name__ == "__main__":
    N_SAMPLES = 10_000
    
    # Recalculate stats
    import polars as pl
    df = pl.read_csv("dev/data/hierarchical_clustered_coords.csv")
    df = df.sample(n=N_SAMPLES, with_replacement=False, seed=42)
    ds = ImageDataset(
        image_paths=df["path"].to_list(),
        output_values=df.select(pl.col("lat"), pl.col("lon"), pl.col("cluster_0")).rows(),
        size=(224, 224)
    )
    
    print("Calculating stats...")
    
    mean, std = calculate_stats(ds, N_SAMPLES)
    print("MEAN = ", mean)
    print("STD  = ", std)