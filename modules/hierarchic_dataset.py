
# The job of the hierarchic dataset is to provide data in hierarchy format.
# We want to do the following
# - retrieve information about the hierarchy levels
# - define a custom sampler for all hierarchy levels
import copy
from typing import Callable, Optional, Tuple, Dict, List, Any, Union
from torch.utils.data import Dataset, Sampler


class HierarchyInformation:
    """Lightweight representation of hierarchy metadata."""

    def __init__(
        self,
        hierarchy_info: Dict[Tuple, List[Tuple]],
        leaf_counts: Optional[Dict[Tuple, int]] = None,
        key_length: Optional[int] = None,
        leaf_offsets: Optional[Dict[Tuple, int]] = None,
    ):
        if not hierarchy_info:
            raise ValueError("hierarchy_info must not be empty")

        self.hierarchy_info = hierarchy_info
        self.key_length = key_length or len(next(iter(hierarchy_info.keys())))
        self.leaf_counts = leaf_counts or {}
        self.leaf_offsets = leaf_offsets or {}
        self._size_cache: Dict[Tuple, int] = {}

    def clone(self) -> "HierarchyInformation":
        """Return a deep-copied hierarchy information object."""
        leaf_offsets = copy.deepcopy(getattr(self, "leaf_offsets", {}))
        return HierarchyInformation(
            copy.deepcopy(self.hierarchy_info),
            copy.deepcopy(self.leaf_counts),
            self.key_length,
            leaf_offsets,
        )

    def get_children(self, level: Tuple) -> List[Tuple]:
        return self.hierarchy_info.get(level, [])

    def is_leaf(self, level: Tuple) -> bool:
        return level in self.leaf_counts

    def get_leaf_size(self, level: Tuple) -> int:
        return self.leaf_counts.get(level, 0)

    def get_size_of_level(self, level: Tuple) -> int:
        """Get the total number of datapoints at and below a given hierarchy level."""
        if level not in self.hierarchy_info and level not in self.leaf_counts:
            raise ValueError(f"Level {level} not found in hierarchy information.")

        if len(level) == 0:
            raise ValueError("Level tuple cannot be empty.")
        if len(level) != self.key_length:
            raise ValueError(
                f"Level tuple length {len(level)} does not match expected length {self.key_length}."
            )

        if level in self._size_cache:
            return self._size_cache[level]

        if self.is_leaf(level):
            size = self.get_leaf_size(level)
        else:
            children = self.get_children(level)
            size = sum(self.get_size_of_level(child) for child in children)

        self._size_cache[level] = size
        return size

    def get_hierarchy_levels(self) -> List[Tuple]:
        return list(self.hierarchy_info.keys())

    def get_level_sizes(self) -> Dict[Tuple, int]:
        return copy.deepcopy(self.leaf_counts)

    def get_leaf_offset(self, level: Tuple) -> int:
        leaf_offsets = getattr(self, "leaf_offsets", {})
        if level not in leaf_offsets:
            raise KeyError(f"Leaf offset not found for level {level}.")
        return leaf_offsets[level]

    def get_leaf_index_range(self, level: Tuple) -> Tuple[int, int]:
        offset = self.get_leaf_offset(level)
        size = self.get_leaf_size(level)
        return offset, offset + size


class HierarchicDataset(HierarchyInformation, Dataset):
    """
    A dataset organized in a hierarchical manner. 
    Note that hierarchy levels must be represented as complete tuples, e.g., (level_0_id, level_1_id, ..., level_n_id).
    level 0 IS THE MOST GRANULAR LEVEL, level n IS THE MOST GENERAL/COARSE/TOP LEVEL/
    """

    def __init__(self, data: Dict[Tuple, Union[List[Any], Dataset]], hierarchy_info: Dict[Tuple, List[Tuple]]):
        """Initialize the HierarchicDataset with data and hierarchy information.
        Args:
            data (Dict[Tuple, Any]): The dataset organized in a hierarchical manner. Keys: Tuple representing hierarchy levels, Values: data samples.
            hierarchy_info (Dict[Tuple, list[Tuple]]): Information about the hierarchy levels. Keys: Tuple representing hierarchy levels, Values: list of tuples indicating all children of this level.
        
        Note: This method does no validation. We expect:
        - data keys are COMPLETE tuples (none have null/missing levels)
        - hierarchy_info keys are nullable and in right-triangle form (e.g., (None, None), (None, 1), (1, 2), etc.)
        """
        self.data = data
        leaf_counts = {level: len(samples) for level, samples in data.items()}
        key_length = len(next(iter(data.keys())))
        self._cumulative_counts = {}
        self._count = 0
        for hpath in self.data.keys():
            self._cumulative_counts[hpath] = self._count
            self._count += len(self.data[hpath])

        super().__init__(
            hierarchy_info,
            leaf_counts=leaf_counts,
            key_length=key_length,
            leaf_offsets=self._cumulative_counts,
        )


    def __len__(self):
        """Get the total number of datapoints in the dataset."""
        return self._count
    
    def __getitem__(self, idx: int) -> Any:
        """Retrieve a datapoint by its global index."""
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range.")
        
        # Find the correct hierarchy path for the given index
        for hpath, start_idx in self._cumulative_counts.items():
            end_idx = start_idx + len(self.data[hpath])
            if start_idx <= idx < end_idx:
                local_idx = idx - start_idx
                sample = self.data[hpath][local_idx]
                if isinstance(sample, tuple):
                    return (*sample, idx)
                return sample, idx
        
        raise IndexError("Index not found in dataset.")
    
    def get_hierarchy_path(self, idx: int) -> Tuple:
        """Get the hierarchy path (tuple) for a given global index."""
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range.")
        
        for hpath, start_idx in self._cumulative_counts.items():
            end_idx = start_idx + len(self.data[hpath])
            if start_idx <= idx < end_idx:
                return hpath
        
        raise IndexError("Index not found in dataset.")
    
    def get_indices_for_level(self, level: Tuple) -> List[int]:
        """Get all global indices that belong to a specific hierarchy level."""
        if level not in self.data:
            return []
        
        start_idx = self._cumulative_counts[level]
        end_idx = start_idx + len(self.data[level])
        return list(range(start_idx, end_idx))
    
    def get_hierarchy_levels(self) -> List[Tuple]:
        """Get all hierarchy levels (keys) in the dataset."""
        return list(self.data.keys())
    
    def get_level_sizes(self) -> Dict[Tuple, int]:
        """Get the size of each hierarchy level."""
        return self.leaf_counts.copy()

    def to_hierarchy_information(self) -> HierarchyInformation:
        """Return a lightweight hierarchy information copy without dataset references."""
        return self.clone()
    
class KeyDefaultDict(dict):
    def __init__(self, default_factory):
        self.default_factory = default_factory

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.default_factory(key)
        return dict.__getitem__(self, key)

class PerLevelSampler(Sampler):
    """
    A sampler that adapts other torch samplers to work with hierarchical datasets.
    For each hierarchy level, it creates a separate sampler using the provided factory function.
    Each time a sample is requested, we traverse down the hierarchy, sampling at each level to get to the next.
    """
    
    def __init__(self, dataset: HierarchicDataset, 
                 sampler_factory: Callable[[List[Any]], Sampler],
                 num_samples: Optional[int] = None,
                 respect_factory_stopiteration: bool = False):
        """
        Args:
            dataset: The HierarchicDataset to sample from. Needed to know hierarchy structure.
            sampler_factory: Factory function to create a sampler for each level.
                             Called with a list of labels (one per item in the level of the hierarchy),
                             and should return a Sampler that can sample among those labels.
            num_samples: Total number of samples to generate per epoch
        """
        self.dataset = dataset
        self.sampler_factory = sampler_factory
        self.num_samples = num_samples or len(dataset)
        self.respect_factory_stopiteration = respect_factory_stopiteration

        # Lazy initialization of samplers using KeyDefaultDict
        self._level_samplers = KeyDefaultDict(self._create_sampler_for_level)
        self._sampler_iters = {}  # Cache iterators for each level
    
    def _create_sampler_for_level(self, level_path: Tuple) -> Sampler:
        """Lazily create a sampler for a given level."""
        # Check if this is a leaf level (actual data indices)
        if level_path in self.dataset.data:
            indices = self.dataset.get_indices_for_level(level_path)
            return self.sampler_factory(indices)
        
        # Otherwise it's a hierarchy level (children to sample from)
        children = self.dataset.hierarchy_info.get(level_path, [])
        if not children:
            raise ValueError(f"No children found for hierarchy level {level_path}.")
        return self.sampler_factory(children)
    
    def _get_sampler_iter(self, level_path: Tuple):
        """Get or create an iterator for a level's sampler."""
        if level_path not in self._sampler_iters:
            self._sampler_iters[level_path] = iter(self._level_samplers[level_path])
        return self._sampler_iters[level_path]
    
    def _sample_hierarchy_path(self, current_level: Tuple) -> Optional[Tuple]:
        """
        Sample a complete hierarchy path from current level to bottom.
        Returns None if all samplers are exhausted (when respect_factory_stopiteration is True).
        """
        for _ in range(self.dataset.key_length):
            none_index = current_level.index(None)
            children = self.dataset.hierarchy_info[current_level]
            sampler_iter = self._get_sampler_iter(current_level)
            
            try:
                child_idx = next(sampler_iter)
            except StopIteration:
                if self.respect_factory_stopiteration:
                    return None
                
                # Recreate iterator and try again
                self._sampler_iters[current_level] = iter(self._level_samplers[current_level])
                child_idx = next(self._sampler_iters[current_level])
            
            # update current_level
            new_level = list(current_level)
            new_level[none_index] = children[child_idx]
            current_level = tuple(new_level)
        
        return current_level
    
    def __iter__(self):
        """Generate num_samples by sampling through the hierarchy."""
        top_level = tuple([None] * self.dataset.key_length)
        
        for _ in range(self.num_samples):
            leaf_path = self._sample_hierarchy_path(top_level)
            
            if leaf_path is None:
                # All samplers exhausted (only when respect_factory_stopiteration is True)
                break
            
            # Use the sampler for the leaf level to sample an index
            leaf_sampler_iter = self._get_sampler_iter(leaf_path)
            
            try:
                yield next(leaf_sampler_iter)
            except StopIteration:
                if self.respect_factory_stopiteration:
                    break
                # Recreate iterator and try again
                self._sampler_iters[leaf_path] = iter(self._level_samplers[leaf_path])
                yield next(self._sampler_iters[leaf_path])
    
    def __len__(self):
        """Return the number of samples per epoch."""
        return self.num_samples


if __name__ == "__main__":
    import polars as pl
    import ast, os
    from image_dataset import ImageDataset
    
    coords_path = "../dev/data/hierarchical_clustered_coords.csv"
    hierarchy_path = "../dev/data/hierarchical_structure.csv"
    
    df = pl.read_csv(coords_path)
    df = df.with_columns([
        (pl.lit("../dev/data/") + pl.col('path')).alias('path')
    ]).filter(
        (pl.col("cluster_0").is_not_null()) & (pl.col("cluster_0") >= 0)
    )
    
    data_dir = os.path.dirname(coords_path)
    
    # Find all cluster columns to determine hierarchy depth
    cluster_cols = [col for col in df.columns if col.startswith('cluster_')]
    num_levels = len(cluster_cols)
    
    # Load hierarchy from CSV
    hierarchy_df = pl.read_csv(hierarchy_path)
    
    hierarchy_df = hierarchy_df.with_columns(
        pl.col('children').map_elements(ast.literal_eval, return_dtype=pl.List(pl.Int64)).alias('children')
    )

    # Build hierarchy_info dict: maps (cluster_N, ..., cluster_0) -> list of children tuples
    print(hierarchy_df)
    hierarchy_info = {
        tuple(
            int(row[rname]) if row[rname] not in (None, '') else None
            for rname in reversed(cluster_cols)
        ): row['children']
        for row in hierarchy_df.iter_rows(named=True)
    }
    
    # Build sub datasets
    def create_data_dict(split_df):
        """maps (cluster_0, cluster_1, ...) -> ImageDataset -> (lat, lon, cluster_0, cluster_1, ...)"""
        paths_dict = {}
        outputs_dict = {}
        
        for row in split_df.iter_rows(named=True):
            key = tuple(
                int(row[rname]) if row[rname] not in (None, '') else None
                for rname in reversed(cluster_cols)
            )
            
            if key not in paths_dict:
                paths_dict[key] = []
                outputs_dict[key] = []
            
            paths_dict[key].append(row['path'])
            outputs_dict[key].append((row['lat'], row['lon']) + key)
        
        assert len(paths_dict) > 0, "No data grouped by hierarchy keys"
        return {k: ImageDataset(paths_dict[k], outputs_dict[k], (224,224)) for k in paths_dict.keys()}
    
    df_data = create_data_dict(df)
    print(next(iter(df_data.items())))
    hd = HierarchicDataset(df_data, hierarchy_info)
    
    # Example usage of HierarchicDataset
    print(f"Total dataset size: {len(hd)}")
    for i in range(5):
        item = hd[i]
        path = hd.get_hierarchy_path(i)
        print(f"Index {i}: Hierarchy Path {path}, Data: {item}")
        
    # Example usage of PerLevelSampler
    from samplers import create_sqrt_sampler
    sampler = PerLevelSampler(hd, create_sqrt_sampler, num_samples=20)
    print("\nSampled indices:")
    for idx in sampler:
        path = hd.get_hierarchy_path(idx)
        print(f"Sampled Index {idx}: Hierarchy Path {path}, Data: {hd[idx]}")