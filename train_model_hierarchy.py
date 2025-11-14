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
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import bitsandbytes as bnb
import argparse
import glob
import re
import gc
from itertools import islice
from functools import partial
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any, List
import ast

# Suppress new loggers
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from modules.image_dataset import ImageDataset
from modules.hierarchic_dataset import HierarchicDataset, PerLevelSampler, HierarchyInformation
from modules.hierarchic_geo_clip_liquid_classifier import HierarchicGeoClassifier
from modules.samplers import create_sqrt_sampler

# Shut down unnecessary logging
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
IMAGE_SIZE = (224, 224)
EARTH_RADIUS_KM = 6371
BOOST_VALUE = 2.0

# Device and optimization settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.enable_mem_efficient_sdp(True)


def load_hierarchical_structure(coords_path: str, centroids_path: str, hierarchy_path: str, train_test_split: float = 0.85) -> Tuple[HierarchicDataset, HierarchicDataset, Dict[int, Tuple[torch.Tensor, ...]], int]:
    """
    Load hierarchical structure from CSV files and create train/test HierarchicDatasets.
    Returns (train_dataset, test_dataset, level_tensors, num_levels).
    """
    assert 0 < train_test_split < 1, f"train_test_split must be between 0 and 1, got {train_test_split}"
    
    # Load coordinates with all cluster levels
    df = pl.read_csv(coords_path)
    data_dir = os.path.dirname(coords_path)
    
    # Find all cluster columns to determine hierarchy depth
    cluster_cols = [col for col in df.columns if col.startswith('cluster_')]
    num_levels = len(cluster_cols)
    assert num_levels > 0, f"No cluster columns found in {coords_path}"
    
    # Add full paths and train/test split
    np.random.seed(42)
    df = df.cast({col: pl.Int64 for col in cluster_cols}, strict=False)
    df = df.with_columns([
        pl.lit(np.random.rand(df.shape[0])).alias("random"),
        (pl.lit(f"{data_dir}/") + pl.col('path')).alias('path')
    ])
    for rname in cluster_cols: # filter out noise points
        df = df.filter(pl.col(rname).is_not_null() & (pl.col(rname) >= 0))
        
    assert len(df) > 0, "No valid data points after filtering noise clusters"
    
    # Load cluster centroids for each level - cast empty strings to null and then to int
    centroids_df = pl.read_csv(centroids_path)
    centroids_df = centroids_df.cast(
        {rname: pl.Int64 for rname in cluster_cols},
        strict=False
    )
    
    # Per-level tensors store (cluster_ids, lat, lon, mean_dist, std_dist)
    level_tensors = {}
    
    for level in range(num_levels):
        cluster_col = f'cluster_{level}'
        
        # Filter to rows for this specific level
        level_df = centroids_df.filter(
            (pl.col(cluster_col).is_not_null()) & (pl.col(cluster_col) >= 0)
        )
        
        # Only get rows where this is the leaf level (all lower levels are None)
        for lower_level in range(level):
            level_df = level_df.filter(pl.col(f'cluster_{lower_level}').is_null())
        
        level_df = level_df.select(
            cluster_col, "lat", "lon", "mean_dist", "std_dist"
        ).sort(cluster_col)
        
        assert len(level_df) > 0, f"No centroids found for level {level}"
        
        # Convert to tensors
        data = level_df.to_numpy()
        level_tensors[level] = (
            torch.tensor(data[:, 0].astype(int)),
            torch.tensor(data[:, 1]),
            torch.tensor(data[:, 2]),
            torch.tensor(data[:, 3]),
            torch.tensor(data[:, 4])
        )
    
    # Load hierarchy from CSV
    hierarchy_df = pl.read_csv(hierarchy_path)
    hierarchy_df = hierarchy_df.cast(
        {rname: pl.Int64 for rname in cluster_cols},
        strict=False
    )
        
    for rname in cluster_cols:
        hierarchy_df = hierarchy_df.filter((pl.col(rname) >= 0) | (pl.col(rname).is_null()))
    
    hierarchy_df = hierarchy_df.with_columns(
        pl.col('children').map_elements(ast.literal_eval, return_dtype=pl.List(pl.Int64)).alias('children')
    )

    # Build hierarchy_info dict: maps (cluster_N, ..., cluster_0) -> list of children tuples
    hierarchy_info = {}
    for row in hierarchy_df.iter_rows(named=True):
        key = tuple(
            int(row[rname]) if row[rname] not in (None, '') else None
            for rname in reversed(cluster_cols)
        )
        children = [
            int(child)
            for child in row['children']
            if child is not None and int(child) >= 0
        ]
        hierarchy_info[key] = children
    assert len(hierarchy_info) > 0, "No hierarchy information loaded"
    assert tuple([None] * num_levels) in hierarchy_info, "Hierarchy info missing the top-level entry (all None). Check hierarchy CSV preprocessing."
    assert hierarchy_info[tuple([None] * num_levels)], "Top-level hierarchy node has no children to sample"

    # Split train/test
    train_df = df.filter(pl.col("random") <= train_test_split)
    test_df = df.filter(pl.col("random") > train_test_split)
    assert len(train_df) > 0, "No training data after split"
    assert len(test_df) > 0, "No test data after split"
    
    def create_data_dict(split_df):
        """maps (cluster_0, cluster_1, ...) -> ImageDataset -> (lat, lon, cluster_0, cluster_1, ...)"""
        paths_dict = {}
        outputs_dict = {}
        
        for row in split_df.iter_rows(named=True):
            key = tuple(
                int(row[rname]) if row[rname] not in (None, '') else None
                for rname in reversed(cluster_cols)
            )

            if key not in hierarchy_info:
                raise ValueError(f"Hierarchy information missing entry for key {key}. Check hierarchy CSV alignment.")
            
            if key not in paths_dict:
                paths_dict[key] = []
                outputs_dict[key] = []
            
            paths_dict[key].append(row['path'])
            outputs_dict[key].append((row['lat'], row['lon']) + key)
        
        assert len(paths_dict) > 0, "No data grouped by hierarchy keys"
        return {k: ImageDataset(paths_dict[k], outputs_dict[k], IMAGE_SIZE) for k in paths_dict.keys()}

    train_data = create_data_dict(train_df)
    test_data = create_data_dict(test_df)
    
    train_dataset = HierarchicDataset(train_data, hierarchy_info)
    test_dataset = HierarchicDataset(test_data, hierarchy_info)
    
    return train_dataset, test_dataset, level_tensors, num_levels


def get_hierarchical_batch_logits(batch: Tuple[torch.Tensor, Any, torch.Tensor], level_tensors: Dict[int, Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]], List[Tuple], torch.Tensor]:
    """
    Convert batch to (images, hierarchical_logits_dict, hierarchy_paths, global_indices).
    hierarchical_logits_dict maps level -> (cluster_ids, target distributions) for that level.
    """
    images, outputs, global_indices = batch
    batch_size = len(outputs)
    assert batch_size > 0, "Empty batch received"
    assert len(level_tensors) > 0, "No level tensors provided"
    assert isinstance(global_indices, torch.Tensor), "Global indices must be provided as a tensor"
    assert len(global_indices) == batch_size, "Global indices size mismatch"
    
    # Extract data from outputs (lat, lon, cluster_0, cluster_1, ...)
    batch_lats = torch.tensor([out[0] for out in outputs], dtype=torch.float32)
    batch_lons = torch.tensor([out[1] for out in outputs], dtype=torch.float32)
    
    assert torch.all(torch.isfinite(batch_lats)), "Invalid latitude values in batch"
    assert torch.all(torch.isfinite(batch_lons)), "Invalid longitude values in batch"
    assert torch.all((batch_lats >= -90) & (batch_lats <= 90)), "Latitude out of range [-90, 90]"
    assert torch.all((batch_lons >= -180) & (batch_lons <= 180)), "Longitude out of range [-180, 180]"
    
    # Extract cluster IDs for each level
    num_levels = len(level_tensors)
    batch_clusters = {}
    hierarchy_paths = []
    
    for i in range(batch_size):
        assert len(outputs[i]) >= 2 + num_levels, f"Output {i} has insufficient data: expected {2 + num_levels}, got {len(outputs[i])}"
        clusters = tuple(int(outputs[i][2 + level]) for level in range(num_levels))
        hierarchy_paths.append(clusters)
        
        for level in range(num_levels):
            if level not in batch_clusters:
                batch_clusters[level] = []
            batch_clusters[level].append(clusters[level])
    
    # Convert to tensors
    for level in range(num_levels):
        batch_clusters[level] = torch.tensor(batch_clusters[level], dtype=torch.int64)
    
    # Compute logits for each level
    hierarchical_logits = {}
    
    for level in range(num_levels):
        # level_tensors[level] stores (cluster_ids, latitudes, longitudes, mean_distances, std_distances)
        cluster_ids, lats, lons, mean_dists, std_dists = level_tensors[level]
        
        assert len(cluster_ids) > 0, f"No clusters for level {level}"
        assert torch.all(mean_dists > 0), f"Invalid mean_dists at level {level}: must be positive"
        std_dists = torch.clamp(std_dists, min=1e-6)
        effective_scale = torch.clamp(mean_dists + std_dists, min=1e-6)
        
        # Haversine distance calculation with broadcasting
        dlat = torch.deg2rad(batch_lats.unsqueeze(1) - lats.unsqueeze(0))
        dlon = torch.deg2rad(batch_lons.unsqueeze(1) - lons.unsqueeze(0))
        
        a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(batch_lats.unsqueeze(1))) * torch.cos(torch.deg2rad(lats.unsqueeze(0))) * torch.sin(dlon/2)**2
        distances = EARTH_RADIUS_KM * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        
        # Create scores (higher for closer distances)
        normalized_distances = distances / effective_scale.unsqueeze(0)
        scores = -torch.log1p(normalized_distances)
        
        # Boost correct cluster
        correct_mask = cluster_ids.unsqueeze(0) == batch_clusters[level].unsqueeze(1)
        scores = torch.where(correct_mask, scores * BOOST_VALUE, scores)
        
        # Convert to distribution
        level_probs = F.softmax(scores, dim=1)
        assert torch.all(torch.isfinite(level_probs)), f"Non-finite logits at level {level}"

        hierarchical_logits[level] = (
            cluster_ids.clone().to(torch.int64),
            level_probs
        )
    
    return images, hierarchical_logits, hierarchy_paths, global_indices


def collate_hierarchical_logits(batch: list, level_tensors: Dict[int, Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], List[Tuple], torch.Tensor]:
    """Collate batch and compute hierarchical logits."""
    assert len(batch) > 0, "Empty batch received in collate function"
    
    valid = []
    indices = []
    for sample in batch:
        if not isinstance(sample, tuple):
            continue
        if len(sample) == 3:
            img, out, idx = sample
        elif len(sample) == 2:
            img, out = sample
            idx = None
        else:
            continue
        if not torch.is_tensor(img) or not isinstance(out, tuple):
            continue
        if idx is None:
            raise ValueError("HierarchicDataset must return global indices for each sample")
        valid.append((img, out))
        indices.append(int(idx))
    
    if not valid:
        raise ValueError(f"No valid samples in batch of size {len(batch)}")
    
    if len(valid) < len(batch):
        print(f"Warning: Filtered out {len(batch) - len(valid)} invalid samples from batch")
    
    images = torch.stack([item[0] for item in valid])
    outputs = [item[1] for item in valid]
    indices_tensor = torch.tensor(indices, dtype=torch.int64)
    
    return get_hierarchical_batch_logits((images, outputs, indices_tensor), level_tensors)


def prepare_hierarchical_data(train_dataset: HierarchicDataset, test_dataset: HierarchicDataset, 
                              level_tensors: Dict[int, Tuple[torch.Tensor, ...]], 
                              batch_size: int, batch_size_test: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and test data loaders with hierarchical sampling."""
    assert len(train_dataset) > 0, "Empty training dataset"
    assert len(test_dataset) > 0, "Empty test dataset"
    assert batch_size > 0, f"Invalid batch_size: {batch_size}"
    assert batch_size_test > 0, f"Invalid batch_size_test: {batch_size_test}"
    
    # Create sampler for hierarchical dataset
    train_sampler = PerLevelSampler(
        train_dataset,
        sampler_factory=create_sqrt_sampler,
        num_samples=len(train_dataset)
    )
    
    # DataLoader configuration
    loader_kwargs = {
        "num_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
        "collate_fn": partial(collate_hierarchical_logits, level_tensors=level_tensors)
    }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, **loader_kwargs)
    
    return train_loader, test_loader


def create_hierarchical_model(args: argparse.Namespace, hierarchy_info: HierarchyInformation) -> HierarchicGeoClassifier:
    """Create hierarchical model with specified architecture and parameters."""
    return HierarchicGeoClassifier(
        hierarchical_structure=hierarchy_info,
        clip_model_name=args.clip_model_name,
        lr=args.learning_rate,
        num_head_dims=args.embed_dim,
        num_hidden_dims=args.num_hidden_dims,
        heads=args.heads,
        depth=args.depth,
        enable_checkpointing=not args.compile,
        device=device,
        dtype=torch.bfloat16
    )


def get_model_filename(batch_count: Optional[int] = None, training: bool = False) -> str:
    """Generate standardized model filename."""
    base = "hierarchic_geo_clip_model"
    if training:
        return f"{base}_training.pth"
    return f"{base}_batch_{batch_count}.pth" if batch_count else f"{base}_final.pth"


def find_latest_model(save_dir: str) -> Tuple[Optional[str], int]:
    """Find the latest model checkpoint based on batch count."""
    # Check for final checkpoint first
    final_path = os.path.join(save_dir, get_model_filename())
    if os.path.exists(final_path):
        return final_path, 0
    
    # Check for training checkpoint next
    training_path = os.path.join(save_dir, get_model_filename(training=True))
    if os.path.exists(training_path):
        return training_path, 0
    
    # Find numbered checkpoints last
    pattern = os.path.join(save_dir, get_model_filename(batch_count="*").replace("*", "[0-9]*"))
    batch_files = [(int(m.group(1)), f) for f in glob.glob(pattern) if (m := re.search(r'batch_(\d+)\.pth', f))]
    
    if not batch_files:
        return None, 0
    
    max_batch, latest_file = max(batch_files)
    return latest_file, max_batch


def load_model_checkpoint(args: argparse.Namespace, hierarchy_info: HierarchyInformation) -> HierarchicGeoClassifier:
    """Load existing checkpoint or create new model."""
    if args.retrain:
        print("--retrain flag set. Starting fresh training.")
        return create_hierarchical_model(args, hierarchy_info)
    
    latest_path, _ = find_latest_model(args.save_dir)
    
    if latest_path and os.path.exists(latest_path):
        model = HierarchicGeoClassifier.load(latest_path)
        if model.hierarchy_information.key_length != hierarchy_info.key_length:
            raise ValueError("Hierarchy mismatch between checkpoint and current dataset metadata.")
        print(f"Loaded model from {latest_path}, resuming from batch {model.total_batches_trained}")
        return model
    
    print("No checkpoint found. Starting fresh training.")
    return create_hierarchical_model(args, hierarchy_info)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description='Train hierarchical geolocation model')
    
    # Data parameters
    parser.add_argument('--centroids_file', type=str, default='dev/data/hierarchical_cluster_centroids.csv')
    parser.add_argument('--coords_file', type=str, default='dev/data/hierarchical_clustered_coords.csv')
    parser.add_argument('--hierarchy_file', type=str, default='dev/data/hierarchical_structure.csv')
    
    # Training parameters
    parser.add_argument('--train_test_split', type=float, default=0.85)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--batch_size_test', type=int, default=24)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--test_frac', type=float, default=0.015)
    parser.add_argument('--max_batches', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=0.0025)
    parser.add_argument('--save_every', type=int, default=-1, help='Save model every N batches (-1 for test intervals only)')
    parser.add_argument('--save_batch_name', action='store_true', help='Save periodic checkpoints with batch number in name')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--retrain', action='store_true', help='Start fresh, ignore existing checkpoints')
    
    # Model architecture
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=2048)
    parser.add_argument('--num_hidden_dims', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--compile', action='store_true', help='Compile model with TorchScript for performance')
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-large-patch14')
    
    return parser.parse_args()


def train_model(model: HierarchicGeoClassifier, train_loader: DataLoader, test_loader: DataLoader, args: argparse.Namespace) -> None:
    """Main training loop with periodic evaluation and checkpointing."""
    os.makedirs(args.save_dir, exist_ok=True)
    losses_file = os.path.join(args.save_dir, "losses_hierarchical.csv")
    
    # If restarting training, replace the losses file
    if model.total_batches_trained == 0 or not os.path.exists(losses_file):
        with open(losses_file, "w") as f:
            f.write("batch_count,lr,loss,test_loss,test_acc\n")
    
    train_transforms = ImageDataset.get_transforms(train=True)
    test_transforms = ImageDataset.get_transforms(train=False)
    
    print(f"\n🚀 Starting hierarchical training on {len(train_loader.dataset):,} images")
    print(f"📊 Model: Hierarchical GeoClip | Depth: {args.depth} | Embed: {args.embed_dim} | Hidden: {args.num_hidden_dims} | Heads: {args.heads}")
    print(f"🔄 Current batch: {model.total_batches_trained:,} / {args.max_batches:,}\n")
    
    pbar = tqdm(desc="Training", initial=model.total_batches_trained, unit="batch", dynamic_ncols=True)
    test_loss = test_acc = float("inf")
    
    try:
        while model.total_batches_trained < args.max_batches:
            for batch in train_loader:
                if args.compile and device.type == 'cuda':
                    torch.compiler.cudagraph_mark_step_begin()
                
                # Efficient GPU transfer
                images = batch[0].to(device, non_blocking=True, dtype=torch.bfloat16)
                hierarchical_logits = {
                    level: (
                        ids.to(device, non_blocking=True, dtype=torch.int64),
                        probs.to(device, non_blocking=True, dtype=torch.bfloat16)
                    )
                    for level, (ids, probs) in batch[1].items()
                }
                hierarchy_paths = batch[2]
                global_indices = batch[3]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model.train_batch((images, hierarchical_logits, global_indices, hierarchy_paths), transforms=train_transforms)
                
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test Acc": f"{test_acc:.4f}"})
                pbar.refresh()
                
                # Periodic checkpointing
                if args.save_every > 0 and model.total_batches_trained % args.save_every == 0:
                    filename = get_model_filename(model.total_batches_trained if args.save_batch_name else None, training=not args.save_batch_name)
                    model.save(os.path.join(args.save_dir, filename))
                    pbar.write(f"\n💾 Saved checkpoint: {filename}")
                
                # Periodic evaluation
                if model.total_batches_trained % args.test_every == 0:
                    test_batches = int(len(test_loader) * args.test_frac)
                    test_loss, test_acc = model.evaluate(
                        tqdm(islice(test_loader, test_batches), desc="Evaluating", total=test_batches, unit="batch"),
                        transforms=test_transforms
                    )
                    model.update_scheduler(test_loss)
                    
                    with open(losses_file, "a") as f:
                        f.write(f"{model.total_batches_trained},{model.get_current_lr()},{loss},{test_loss},{test_acc}\n")
                    
                    if args.save_every < 0:  # Default behavior: save on test
                        model.save(os.path.join(args.save_dir, get_model_filename(training=True)))
                    
                    pbar.write(f"\n📈 Batch {model.total_batches_trained:,}: Loss={loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, LR={model.get_current_lr():.2e}\n")
                
                # "Fix" memory leak
                if model.total_batches_trained % 10 == 0:
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    gc.collect()
                
                pbar.refresh()
                if model.total_batches_trained >= args.max_batches:
                    break
                    
    except KeyboardInterrupt:
        pbar.write("\n\n⚠️  Training interrupted by user")
    finally:
        pbar.close()
        final_path = os.path.join(args.save_dir, get_model_filename())
        model.save(final_path)
        print(f"\n✅ Training complete! Final model saved to {final_path}")
        print(f"📊 Total batches trained: {model.total_batches_trained:,}")


def main():
    """Main entry point with complete training pipeline."""
    args = parse_arguments()
    
    print(f"\n🌍 Loading hierarchical structure...")
    
    # Load hierarchical structure from CSV
    train_dataset, test_dataset, level_tensors, num_levels = load_hierarchical_structure(
        args.coords_file, 
        args.centroids_file,
        args.hierarchy_file,
        args.train_test_split
    )
    
    print(f"📊 Loaded {num_levels} hierarchy levels")
    print(f"📊 Train dataset: {len(train_dataset):,} samples")
    print(f"📊 Test dataset: {len(test_dataset):,} samples")
    
    hierarchy_info = train_dataset.to_hierarchy_information()

    # Prepare data loaders
    train_loader, test_loader = prepare_hierarchical_data(
        train_dataset, test_dataset, level_tensors,
        args.batch_size, args.batch_size_test
    )
    
    # Load or create model
    model = load_model_checkpoint(args, hierarchy_info)
    model.send_to_device(device, dtype=torch.bfloat16)
    
    if args.compile:
        model.compile(
            fullgraph=False,
            dynamic=False,
            backend='inductor',
            mode='default'
        )
    
    # Train the model
    train_model(model, train_loader, test_loader, args)


if __name__ == "__main__":
    main()