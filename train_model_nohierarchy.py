import polars as pl
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import glob
import re

from modules.image_dataset import ImageDataset
from modules.visiontranformer_model import VisionTransformerBase
from modules.geo_clip_liquid_classifier import GeoLiquidClipModel
from modules.geo_clip_frozen_classifier import GeoFrozenClipModel
from modules.geo_vt_classifier import GeoVTModel

from itertools import islice
from tqdm import tqdm
import os
import gc

IMAGE_SIZE = (224, 224)  # must be consistent with vt and clip models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable automatic mixed precision with bfloat16
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)

def create_cluster_mapping(centroids_path):
    """Convert cluster stats to efficient mapping format for fast lookups."""
    # Convert to dictionary for fast lookups
    cluster_stats = pl.read_csv(centroids_path).select(
        pl.col("cluster_0"), pl.col("lat"), pl.col("lon"), pl.col("avg_dist")
    ).filter(
        pl.col("cluster_0").is_not_null() & pl.col("cluster_0") >= 0 # remove outlier cluster
    ).sort(pl.col("cluster_0")) # make sure logits are in order
    
    cluster_dict = {}
    for row in cluster_stats.iter_rows(named=True):
        cluster_dict[row['cluster_0']] = (row['lat'], row['lon'], row['avg_dist'])
    
    # Create PyTorch tensors for batch operations
    clusters = torch.tensor([k for k in cluster_dict.keys()])
    lats = torch.tensor([v[0] for v in cluster_dict.values()])
    lons = torch.tensor([v[1] for v in cluster_dict.values()])
    avg_dists = torch.tensor([v[2] for v in cluster_dict.values()])
    
    return clusters, lats, lons, avg_dists

# Global variable to store cluster tensors
CLUSTER_TENSORS = None

def get_batch_logits(batch, boost_value = 10):
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
    logits = torch.where(correct_cluster_mask, logits + boost_value, logits)  # Add small boost to correct clusters
    
    return images, logits

def collate_with_logits(batch):
    """Custom collate function for DataLoaders that applies get_batch_logits transformation"""
    # Default collate to get standard batch format
    images = torch.stack([item[0] for item in batch])
    outputs = torch.tensor([item[1] for item in batch])
    
    # Apply the logits transformation
    images, logits = get_batch_logits((images, outputs))
    
    return images, logits

# Prepare data
def prepare_data(coords_file, train_test_split, batch_size, batch_size_test):
    # Load data
    df = pl.read_csv(coords_file)
    
    # Extract directory from coords file path
    data_dir = os.path.dirname(coords_file)
    
    # Add random column for splitting
    np.random.seed(42)  # For reproducibility
    df = df.with_columns(pl.lit(np.random.rand(df.shape[0])).alias("random")).filter(
        pl.col("cluster_0").is_not_null() & pl.col("cluster_0") >= 0 # remove outlier cluster
    ).with_columns(
        (pl.lit(data_dir + '/') + pl.col('path')).alias('path')  # Add path prefix
    )
    
    # Split based on random value
    train_df = df.filter(pl.col("random") <= train_test_split).drop("random")
    test_df  = df.filter(pl.col("random") > train_test_split).drop("random")
    
    train_input_values = train_df.select(pl.col("path")).to_series().to_list()
    test_input_values  = test_df.select(pl.col("path")).to_series().to_list()

    train_output_values = train_df.select(pl.col("lat"), pl.col("lon"), pl.col("cluster_0")).rows()
    test_output_values  = test_df.select(pl.col("lat"), pl.col("lon"), pl.col("cluster_0")).rows()

    # Create datasets
    train_dataset = ImageDataset(image_paths=train_input_values, output_values=train_output_values, size=IMAGE_SIZE)    
    test_dataset  = ImageDataset(image_paths=test_input_values, output_values=test_output_values, size=IMAGE_SIZE)
    
    # Create data loaders with GPU pinning
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_with_logits
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size_test, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_with_logits
    )
    
    num_clusters = len(df.select(pl.col("cluster_0")).unique())
    
    return train_loader, test_loader, num_clusters

def create_model(args, num_clusters):
    """Create model based on arguments"""
    model_params = {
        'lr': args.learning_rate,
        'num_classes': num_clusters,
        'num_head_dims': args.embed_dim,
        'num_hidden_dims': args.num_hidden_dims,
        'heads': args.heads,
        'depth': args.depth,
        'device': device,
        'dtype': torch.bfloat16
    }
    
    if args.base_model == "clip":
        model_params['clip_model_name'] = args.clip_model_name
        model = GeoFrozenClipModel(**model_params) if args.freeze_clip else GeoLiquidClipModel(**model_params)
    else:  # vt
        try:
            model_params['vt_base'] = VisionTransformerBase[args.vt_base]
        except KeyError:
            raise ValueError(f"Invalid Vision Transformer base model: {args.vt_base}")
        
        model = GeoVTModel(**model_params)
    
    return model

def get_model_filename(base_model, frozen, batch_count=None, training=False):
    """Generate model filename based on architecture and batch count"""
    frozen_str = "frozen" if frozen else "liquid"
    base_name = f"{base_model}_{frozen_str}_model"
    
    if training:
        return f"{base_name}_training.pth"
    elif batch_count is not None:
        return f"{base_name}_batch_{batch_count}.pth"
    else:
        return f"{base_name}_final.pth"

def find_latest_model(save_dir, base_model, frozen):
    """Find the latest model checkpoint based on batch count"""
    # First check for training checkpoint (highest priority)
    training_path = os.path.join(save_dir, get_model_filename(base_model, frozen, training=True))
    if os.path.exists(training_path):
        return training_path, 0  # We don't know the batch count from training files
    
    # Then look for batch checkpoints
    pattern = os.path.join(save_dir, get_model_filename(base_model, frozen, batch_count="*").replace("*", "[0-9]*"))
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None, 0
    
    # Extract batch counts and find max
    batch_files = []
    for f in model_files:
        match = re.search(r'batch_(\d+)\.pth', f)
        if match:
            batch_files.append((int(match.group(1)), f))
    
    if not batch_files:
        return None, 0
    
    # Get file with highest batch count
    max_batch, latest_file = max(batch_files, key=lambda x: x[0])
    return latest_file, max_batch

def load_model_checkpoint(args, num_clusters):
    """Load existing model checkpoint or create new model"""
    if args.retrain:
        print("--retrain flag set. Starting fresh training.")
        return create_model(args, num_clusters), 0
    
    # Try to find and load existing model
    latest_path, start_batch = find_latest_model(args.save_dir, args.base_model, args.freeze_clip)
    
    if latest_path and os.path.exists(latest_path):
        print(f"Model found at {latest_path}, resuming training" + (f" from batch {start_batch}" if start_batch > 0 else ""))
        
        # Load the appropriate model class
        model_classes = {
            ('clip', True): GeoFrozenClipModel,
            ('clip', False): GeoLiquidClipModel,
            ('vt', False): GeoVTModel,
            ('vt', True): GeoVTModel
        }
        
        model_class = model_classes[(args.base_model, args.freeze_clip)]
        return model_class.load(latest_path), start_batch
    
    print("No model found. Starting fresh training.")
    return create_model(args, num_clusters), 0

# Main function
def main():
    parser = argparse.ArgumentParser(description='Train geolocation models')
    
    # Data parameters
    parser.add_argument('--centroids_file', type=str, default='dev/data/hierarchical_cluster_centroids.csv', 
                        help='Path to centroids CSV file')
    parser.add_argument('--coords_file', type=str, default='dev/data/hierarchical_clustered_coords.csv', 
                        help='Path to coordinates CSV file')
    
    # Training parameters
    parser.add_argument('--train_test_split', type=float, default=0.85, help='Train/test split ratio')
    parser.add_argument('--batch_size', type=int, default=12, help='Training batch size')
    parser.add_argument('--batch_size_test', type=int, default=24, help='Test batch size')
    parser.add_argument('--test_every', type=int, default=1000, help='Test every N batches')
    parser.add_argument('--test_frac', type=float, default=0.015, help='Fraction of test set to use each test')
    parser.add_argument('--max_batches', type=int, default=100000, help='Maximum number of batches to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=-1, help='Save model every N batches (-1 for default behavior)')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--retrain', action='store_true', help='Start training from scratch, ignore existing models')
    
    # Model architecture parameters
    parser.add_argument('--base_model', type=str, default='clip', 
                        choices=['clip', 'vt'],
                        help='Base model type')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the model')
    parser.add_argument('--embed_dim', type=int, default=2048, help='Embedding dimension (projection from base model)')
    parser.add_argument('--num_hidden_dims', type=int, default=8192, help='Hidden dimension size')
    parser.add_argument('--heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--freeze_clip', action='store_true', help='Freeze CLIP backbone (use frozen instead of liquid)')
    
    # Model-specific parameters
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')
    parser.add_argument('--vt_base', type=str, default='VIT_B_32', help='Vision Transformer base (for vt model)')
    
    args = parser.parse_args()
    
    # Initialize cluster tensors globally
    global CLUSTER_TENSORS
    CLUSTER_TENSORS = create_cluster_mapping(args.centroids_file)
    
    # Prepare data
    train_loader, test_loader, num_clusters = prepare_data(
        args.coords_file,
        args.train_test_split, 
        args.batch_size, 
        args.batch_size_test
    )
    
    # Get transforms from dataset
    train_transforms = ImageDataset.get_transforms(train=True)
    test_transforms  = ImageDataset.get_transforms(train=False)
    
    print(f"Initializing model... ({num_clusters} clusters)")
    
    # Load model checkpoint or create new
    model, start_batch = load_model_checkpoint(args, num_clusters)
    model.send_to_device(device, dtype=torch.bfloat16)  # Use bfloat16 for training    
    
    # Prep filesystem
    os.makedirs(args.save_dir, exist_ok=True)
    losses_file = os.path.join(args.save_dir, "losses.csv")
    
    # Create or append to losses file
    if start_batch == 0:
        with open(losses_file, "w") as f:
            f.write("batch_count,lr,loss,test_loss,test_acc\n")
    
    model_type = f"{args.base_model}_{'frozen' if args.freeze_clip else 'liquid'}"
    print(f"Starting training...")
    print(f"Training on {len(train_loader.dataset)} images")
    print(f"Model: {model_type}, Depth: {args.depth}, Embed dim: {args.embed_dim}, Hidden dims: {args.num_hidden_dims}, Heads: {args.heads}")
    if start_batch > 0:
        print(f"Resuming from batch {start_batch}")
    
    try:
        pbar = tqdm(desc="Training", unit="batch", initial=start_batch)
        test_loss, test_acc = float("inf"), float("inf")
        batch_count = start_batch
        
        while batch_count < args.max_batches:
            for batch in train_loader:
                # Move batch to GPU and convert to bfloat16
                batch = [io.to(device, non_blocking=True, dtype=torch.bfloat16) for io in batch]
                gc.collect()
                
                # Train model on batch
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model.train_batch(batch, transforms=train_transforms)
                
                batch_count += 1
                pbar.update(1)
                
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test Acc": f"{test_acc:.4f}"})
                
                # Save model at intervals if specified
                if args.save_every > 0 and batch_count % args.save_every == 0:
                    save_path = os.path.join(args.save_dir, 
                                           get_model_filename(args.base_model, args.freeze_clip, batch_count))
                    model.save(save_path)
                    print(f"\nSaved checkpoint at batch {batch_count}")
                
                # Test model at intervals
                if batch_count % args.test_every == 0:
                    # Split test loader by max test batches for this test iteration
                    test_batches = int(len(test_loader) * args.test_frac)
                    test_subset = islice(test_loader, test_batches)
                    
                    # Run test
                    test_loss, test_acc = model.evaluate(tqdm(test_subset, desc="Testing", unit="batch", total=test_batches), transforms=test_transforms)
                    model.update_scheduler(test_loss)
                    with open(losses_file, "a") as f:
                        f.write(f"{batch_count},{model.get_current_lr()},{loss},{test_loss},{test_acc}\n")
                    
                    # Save training checkpoint when save_every is negative
                    if args.save_every < 0:
                        save_path = os.path.join(args.save_dir, get_model_filename(args.base_model, args.freeze_clip, training=True))
                        model.save(save_path)
                    
                    print('\n')
                    
                if batch_count >= args.max_batches:
                    break
                    
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
    finally:
        pbar.close()
        final_path = os.path.join(args.save_dir, get_model_filename(args.base_model, args.freeze_clip))
        model.save(final_path)
        print(f"Final model saved to {final_path}")
    
    print("Training complete.")
    
if __name__ == "__main__":
    main()