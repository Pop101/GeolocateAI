import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from modules.image_dataset import ImageDataset
# from modules.visiontranformer_model import VisionTransformerModel
from modules.geo_clip_model import GeoClipModel

import itertools
from tqdm import tqdm
import os
import shutil
import gc

CENTROIDS_PATH = "dev/data/hierarchical_cluster_centroids.csv"
DATA_PATH = "dev/data/hierarchical_clustered_coords.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable automatic mixed precision with bfloat16
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def create_cluster_mapping():
    """Convert cluster stats to efficient mapping format for fast lookups."""
    # Convert to dictionary for fast lookups
    cluster_stats = pl.read_csv(CENTROIDS_PATH).select(
        pl.col("cluster_0"), pl.col("lat"), pl.col("lon"), pl.col("avg_dist")
    ).filter(
        pl.col("cluster_0").is_not_null() & pl.col("cluster_0") >= 0 # remove outlier cluster
    ).sort(pl.col("cluster_0")) # make sure logits are in order
    
    cluster_dict = {}
    for row in cluster_stats.iter_rows(named=True):
        cluster_dict[row['cluster_0']] = (row['lat'], row['lon'], row['avg_dist'])
    
    # Create PyTorch tensors for batch operations
    clusters = torch.tensor([k for k in cluster_dict.keys()], device=device)
    lats = torch.tensor([v[0] for v in cluster_dict.values()], device=device)
    lons = torch.tensor([v[1] for v in cluster_dict.values()], device=device)
    avg_dists = torch.tensor([v[2] for v in cluster_dict.values()], device=device)
    
    return clusters, lats, lons, avg_dists

# Create the mapping at module level for reuse
CLUSTER_TENSORS = create_cluster_mapping()

def get_batch_logits(batch):
    """
    Converts a batch (images, [lat, lon, cluster]) to 
    batch (images, logits), scoring based on distance to cluster centroid
    """
    
    images, outputs = batch
    
    # Outputs is in format [lat, lon, cluster]
    batch_lats, batch_lons, batch_clusters = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    
    # Get cluster tensors
    clusters, lats, lons, avg_dists = CLUSTER_TENSORS
    
    # Convert to radians for Haversine formula
    batch_lats_rad = torch.deg2rad(batch_lats)
    batch_lons_rad = torch.deg2rad(batch_lons)
    lats_rad = torch.deg2rad(lats)
    lons_rad = torch.deg2rad(lons)
    
    # Calculate Haversine distance
    # Using broadcasting for efficient computation
    dlat = batch_lats_rad.unsqueeze(1) - lats_rad.unsqueeze(0)  # [batch_size, n_clusters]
    dlon = batch_lons_rad.unsqueeze(1) - lons_rad.unsqueeze(0)  # [batch_size, n_clusters]
    
    # Haversine formula
    a = torch.sin(dlat/2)**2 + torch.cos(batch_lats_rad.unsqueeze(1)) * torch.cos(lats_rad.unsqueeze(0)) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    distances = 6371 * c  # Earth's radius in km * c
    
    # Calculate z-scores (distance / avg_dist for each cluster)
    z_scores = distances / avg_dists.unsqueeze(0)  # [batch_size, n_clusters]
    
    # Convert to logits (negative z-scores so lower distances = higher scores)
    logits = -z_scores
    
    # Ensure the correct cluster has the highest logit
    correct_cluster_mask = (clusters.unsqueeze(0) == batch_clusters.unsqueeze(1))  # [batch_size, n_clusters]
    logits = torch.where(correct_cluster_mask, logits + 1.0, logits)  # Add small boost to correct clusters
    
    # Normalize logits to [0,1] range
    min_logits = logits.min(dim=1, keepdim=True)[0]  # [batch_size, 1]
    max_logits = logits.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    logits = (logits - min_logits) / (max_logits - min_logits)  # [batch_size, n_clusters]
    
    return images, logits

# Prepare data
def prepare_data(split_ratio=0.85):
    # Load data
    df = pl.read_csv(DATA_PATH)
    
    # Add random column for splitting
    np.random.seed(42)  # For reproducibility
    df = df.with_columns(pl.lit(np.random.rand(df.shape[0])).alias("random")).filter(
        pl.col("cluster_0").is_not_null() & pl.col("cluster_0") >= 0 # remove outlier cluster
    )
    
    # Split based on random value
    train_df = df.filter(pl.col("random") <= split_ratio).drop("random")
    test_df = df.filter(pl.col("random") > split_ratio).drop("random")
    
    train_input_values = train_df.select(pl.col("path")).to_series().to_list()
    test_input_values = test_df.select(pl.col("path")).to_series().to_list()

    train_output_values = train_df.select(pl.col("lat"), pl.col("lon"), pl.col("cluster_0")).rows()
    test_output_values = test_df.select(pl.col("lat"), pl.col("lon"), pl.col("cluster_0")).rows()

    # Create datasets
    train_dataset = ImageDataset(image_paths=train_input_values, output_values=train_output_values, size=IMAGE_SIZE)    
    test_dataset  = ImageDataset(image_paths=test_input_values, output_values=test_output_values, size=IMAGE_SIZE)
    
    # Create data loaders with GPU pinning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,  # Enable GPU pinning
        persistent_workers=True,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # Enable GPU pinning
        persistent_workers=True,
    )
    
    num_clusters = len(df.select(pl.col("cluster_0")).unique())
    
    return train_loader, test_loader, num_clusters

# Main function
def main():    
    
    # Prepare data
    train_loader, test_loader, num_clusters = prepare_data()
    
    # Get transforms from dataset
    train_transforms = ImageDataset.get_transforms(train=True)
    test_transforms  = ImageDataset.get_transforms(train=False)
    
    model = GeoClipModel(num_classes=num_clusters)
    model.send_to_device(device)
    
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Prep filesystem
    os.makedirs("models", exist_ok=True)
    with open("models/losses.csv", "w") as f:
        f.write("batch_count,lr,loss,test_loss,test_acc\n")
        
    print("Starting training...")
    print(f"Training on {len(train_loader.dataset)} images")
    try:
        pbar = tqdm(desc="Training", unit="batch")
        test_loss, test_acc = float("inf"), float("inf")
        batch_count = 0
        
        while batch_count < 25_000:
            for batch in train_loader:
                # Move batch to GPU and convert to bfloat16
                batch = [b.to(device, non_blocking=True) for b in batch]
                batch = get_batch_logits(batch)
                gc.collect()
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model.train_batch(batch, transforms=train_transforms)
                
                batch_count += 1
                pbar.update(1)
                
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test Acc": f"{test_acc:.4f}"})
                if batch_count % 100 == 0:
                    test_loss, test_acc = model.evaluate(tqdm(test_loader, desc="Testing", unit="batch"), transforms=test_transforms)
                    model.update_scheduler(test_loss)
                    with open("models/losses.csv", "a") as f:
                        f.write(f"{batch_count},{model.get_current_lr()},{loss},{test_loss},{test_acc}\n")
                    print('\n')
 
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
    finally:
        pbar.close()
        model.save("models/image_rating_model_final.pth")
    
    print("Training complete.")
    
if __name__ == "__main__":
    main()