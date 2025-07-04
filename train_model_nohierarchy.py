import os
import logging
import warnings

# Suppress all warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

# Now do the actual imports
import polars as pl
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
import argparse
import glob
import re
import gc
from itertools import islice
from functools import partial
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any

# Suppress new loggers
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from modules.image_dataset import ImageDataset
from modules.visiontranformer_model import VisionTransformerBase
from modules.geo_clip_liquid_classifier import GeoLiquidClipModel
from modules.geo_clip_frozen_classifier import GeoFrozenClipModel
from modules.geo_vt_classifier import GeoVTModel

# Shut down unnecessary logging
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
IMAGE_SIZE = (224, 224)
EARTH_RADIUS_KM = 6371
BOOST_VALUE = 10

# Device and optimization settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)


def create_cluster_mapping(centroids_path: str) -> Tuple[torch.Tensor, ...]:
    """Convert cluster stats to efficient tensor format for fast GPU lookups."""
    cluster_stats = pl.read_csv(
        centroids_path
    ).select(
        "cluster_0", "lat", "lon", "avg_dist"
    ).filter(
        (pl.col("cluster_0").is_not_null()) & (pl.col("cluster_0") >= 0)
    ).sort("cluster_0")
    
    # Direct to tensors for GPU efficiency
    data = cluster_stats.to_numpy()
    return (torch.tensor(data[:, 0].astype(int)), torch.tensor(data[:, 1]), torch.tensor(data[:, 2]), torch.tensor(data[:, 3]))


def get_batch_logits(batch: Tuple[torch.Tensor, torch.Tensor], cluster_tensors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (images, [lat, lon, cluster]) to (images, logits) using Haversine distance scoring."""
    images, outputs = batch
    batch_lats, batch_lons, batch_clusters = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    clusters, lats, lons, avg_dists = cluster_tensors

    # Haversine distance calculation with broadcasting
    dlat = torch.deg2rad(batch_lats.unsqueeze(1) - lats.unsqueeze(0))
    dlon = torch.deg2rad(batch_lons.unsqueeze(1) - lons.unsqueeze(0))
    
    a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(batch_lats.unsqueeze(1))) * torch.cos(torch.deg2rad(lats.unsqueeze(0))) * torch.sin(dlon/2)**2
    distances = EARTH_RADIUS_KM * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    
    # Convert to logits via z-scores
    logits = -distances / avg_dists.unsqueeze(0)
    
    # Boost correct clusters
    correct_mask = clusters.unsqueeze(0) == batch_clusters.unsqueeze(1)
    return images, torch.where(correct_mask, logits + BOOST_VALUE, logits)


def collate_with_logits(batch: list, cluster_tensors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function that applies logits transformation."""
    images = torch.stack([item[0] for item in batch])
    outputs = torch.tensor([item[1] for item in batch])
    return get_batch_logits((images, outputs), cluster_tensors)


def prepare_data(coords_file: str, train_test_split: float, batch_size: int, batch_size_test: int, cluster_tensors: Tuple[torch.Tensor, ...]) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare train and test data loaders with efficient GPU settings."""
    df = pl.read_csv(coords_file)
    data_dir = os.path.dirname(coords_file)
    
    # Perform test / train split, filtering out noise clusters
    np.random.seed(42)
    df = df.with_columns([
        pl.lit(np.random.rand(df.shape[0])).alias("random"),
        (pl.lit(f"{data_dir}/") + pl.col('path')).alias('path')
    ]).filter(
        (pl.col("cluster_0").is_not_null()) & (pl.col("cluster_0") >= 0)
    )
    
    train_df = df.filter(pl.col("random") <= train_test_split).drop("random")
    test_df = df.filter(pl.col("random") > train_test_split).drop("random")
    
    # Create datasets
    train_dataset = ImageDataset(train_df["path"].to_list(), train_df.select("lat", "lon", "cluster_0").rows(), IMAGE_SIZE)
    test_dataset = ImageDataset(test_df["path"].to_list(), test_df.select("lat", "lon", "cluster_0").rows(), IMAGE_SIZE)
    
    # Set up DataLoader with efficient GPU settings
    loader_kwargs = {"num_workers": 4, "pin_memory": True, "persistent_workers": True, "collate_fn": partial(collate_with_logits, cluster_tensors=cluster_tensors)}
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs),     # Train DataLoader
        DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, **loader_kwargs), # Test DataLoader
        df["cluster_0"].n_unique()                                                           # Number of clusters
    )


def create_model(args: argparse.Namespace, num_clusters: int) -> Any:
    """Create model with specified architecture and parameters."""
    base_params = {"lr": args.learning_rate, "num_classes": num_clusters, "num_head_dims": args.embed_dim, "num_hidden_dims": args.num_hidden_dims, "heads": args.heads, "depth": args.depth, "device": device, "dtype": torch.bfloat16}
    
    if args.base_model == "clip":
        base_params["clip_model_name"] = args.clip_model_name
        return GeoFrozenClipModel(**base_params) if args.freeze_clip else GeoLiquidClipModel(**base_params)
    
    base_params["vt_base"] = VisionTransformerBase[args.vt_base]
    return GeoVTModel(**base_params)


def get_model_filename(base_model: str, frozen: bool, batch_count: Optional[int] = None, training: bool = False) -> str:
    """Generate standardized model filename."""
    base = f"{base_model}_{'frozen' if frozen else 'liquid'}_model"
    if training:
        return f"{base}_training.pth"
    return f"{base}_batch_{batch_count}.pth" if batch_count else f"{base}_final.pth"


def find_latest_model(save_dir: str, base_model: str, frozen: bool) -> Tuple[Optional[str], int]:
    """Find the latest model checkpoint based on batch count."""
    # Check for final checkpoint first
    final_path = os.path.join(save_dir, get_model_filename(base_model, frozen))
    if os.path.exists(final_path):
        return final_path, 0
    
    # Check for training checkpoint next
    training_path = os.path.join(save_dir, get_model_filename(base_model, frozen, training=True))
    if os.path.exists(training_path):
        return training_path, 0
    
    # Find numbered checkpoints last
    pattern = os.path.join(save_dir, get_model_filename(base_model, frozen, batch_count="*").replace("*", "[0-9]*"))
    batch_files = [(int(m.group(1)), f) for f in glob.glob(pattern) if (m := re.search(r'batch_(\d+)\.pth', f))]
    
    if not batch_files:
        return None, 0
    
    max_batch, latest_file = max(batch_files)
    return latest_file, max_batch


def load_model_checkpoint(args: argparse.Namespace, num_clusters: int) -> Any:
    """Load existing checkpoint or create new model."""
    if args.retrain:
        print("--retrain flag set. Starting fresh training.")
        return create_model(args, num_clusters)
    
    latest_path, _ = find_latest_model(args.save_dir, args.base_model, args.freeze_clip)
    
    if latest_path and os.path.exists(latest_path):
        model_class = {('clip', True): GeoFrozenClipModel, ('clip', False): GeoLiquidClipModel, ('vt', False): GeoVTModel, ('vt', True): GeoVTModel}[(args.base_model, args.freeze_clip)]
        model = model_class.load(latest_path)
        print(f"Loaded model from {latest_path}, resuming from batch {model.total_batches_trained}")
        return model
    
    print("No checkpoint found. Starting fresh training.")
    return create_model(args, num_clusters)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description='Train geolocation models with style')
    
    # Data parameters
    parser.add_argument('--centroids_file', type=str, default='dev/data/hierarchical_cluster_centroids.csv')
    parser.add_argument('--coords_file', type=str, default='dev/data/hierarchical_clustered_coords.csv')
    
    # Training parameters
    parser.add_argument('--train_test_split', type=float, default=0.85)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--batch_size_test', type=int, default=24)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--test_frac', type=float, default=0.015)
    parser.add_argument('--max_batches', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_every', type=int, default=-1, help='Save model every N batches (-1 for test intervals only)')
    parser.add_argument('--save_batch_name', action='store_true', help='Save periodic checkpoints with batch number in name instead of _training suffix (default)')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--retrain', action='store_true', help='Start fresh, ignore existing checkpoints')
    
    # Model architecture
    parser.add_argument('--base_model', type=str, default='clip', choices=['clip', 'vt'])
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=2048)
    parser.add_argument('--num_hidden_dims', type=int, default=8192)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--freeze_clip', action='store_true', help='Freeze CLIP backbone')
    
    # Model-specific
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--vt_base', type=str, default='VIT_B_32')
    
    return parser.parse_args()


def train_model(model: Any, train_loader: DataLoader, test_loader: DataLoader, args: argparse.Namespace) -> None:
    """Main training loop with periodic evaluation and checkpointing."""
    os.makedirs(args.save_dir, exist_ok=True)
    losses_file = os.path.join(args.save_dir, "losses.csv")
    
    if model.total_batches_trained == 0:
        with open(losses_file, "w") as f:
            f.write("batch_count,lr,loss,test_loss,test_acc\n")
    
    train_transforms = ImageDataset.get_transforms(train=True)
    test_transforms = ImageDataset.get_transforms(train=False)
    
    print(f"\n🚀 Starting training on {len(train_loader.dataset):,} images")
    print(f"📊 Model: {args.base_model}_{'frozen' if args.freeze_clip else 'liquid'} | Depth: {args.depth} | Embed: {args.embed_dim} | Hidden: {args.num_hidden_dims} | Heads: {args.heads}")
    print(f"🔄 Current batch: {model.total_batches_trained:,} / {args.max_batches:,}\n")
    
    pbar = tqdm(desc="Training", initial=model.total_batches_trained)
    test_loss = test_acc = float("inf")
    
    try:
        while model.total_batches_trained < args.max_batches:
            for batch in train_loader:
                # Efficient GPU transfer
                batch = [b.to(device, non_blocking=True, dtype=torch.bfloat16) for b in batch]
                gc.collect()
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model.train_batch(batch, transforms=train_transforms)
                
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test Acc": f"{test_acc:.4f}"})
                
                # Periodic checkpointing
                if args.save_every > 0 and model.total_batches_trained % args.save_every == 0:
                    filename = get_model_filename(args.base_model, args.freeze_clip, model.total_batches_trained if args.save_batch_name else None, training=not args.save_batch_name)
                    model.save(os.path.join(args.save_dir, filename))
                    print(f"\n💾 Saved checkpoint: {filename}")
                
                # Periodic evaluation
                if model.total_batches_trained % args.test_every == 0:
                    test_batches = int(len(test_loader) * args.test_frac)
                    test_loss, test_acc = model.evaluate(tqdm(islice(test_loader, test_batches), desc="Evaluating", total=test_batches), transforms=test_transforms)
                    model.update_scheduler(test_loss)
                    
                    with open(losses_file, "a") as f:
                        f.write(f"{model.total_batches_trained},{model.get_current_lr()},{loss},{test_loss},{test_acc}\n")
                    
                    if args.save_every < 0:  # Default behavior: save on test
                        model.save(os.path.join(args.save_dir, get_model_filename(args.base_model, args.freeze_clip, training=True)))
                    
                    print(f"\n📈 Batch {model.total_batches_trained:,}: Loss={loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, LR={model.get_current_lr():.2e}\n")
                
                if model.total_batches_trained >= args.max_batches:
                    break
                    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    finally:
        pbar.close()
        final_path = os.path.join(args.save_dir, get_model_filename(args.base_model, args.freeze_clip))
        model.save(final_path)
        print(f"\n✅ Training complete! Final model saved to {final_path}")
        print(f"📊 Total batches trained: {model.total_batches_trained:,}")


def main():
    """Main entry point with complete training pipeline."""
    args = parse_arguments()
    
    # Initialize data pipeline
    cluster_tensors = create_cluster_mapping(args.centroids_file)
    train_loader, test_loader, num_clusters = prepare_data(args.coords_file, args.train_test_split, args.batch_size, args.batch_size_test, cluster_tensors)
    
    print(f"\n🌍 Initializing geolocation model with {num_clusters:,} clusters...")
    
    # Load or create model
    model = load_model_checkpoint(args, num_clusters)
    model.send_to_device(device, dtype=torch.bfloat16)
    
    # Train the model
    train_model(model, train_loader, test_loader, args)


if __name__ == "__main__":
    main()