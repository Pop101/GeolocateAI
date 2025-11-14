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
from modules.hierarchic_dataset import HierarchicDataset, HierarchyInformation

from typing import Dict, Tuple, List
import warnings

class HierarchicGeoClassifier:
    def __init__(self, lr=0.001, hierarchical_structure: HierarchyInformation = None, num_head_dims=1024*2, num_hidden_dims=1024*8, heads=32, depth=8, enable_checkpointing=False, clip_model_name="openai/clip-vit-large-patch14", output_type=ClipOutput.POOLER_OUTPUT, device=None, dtype=torch.float32):   
        self.device = device
        self.dtype = dtype
        if hierarchical_structure is None:
            raise ValueError("hierarchical_structure must be provided")

        if isinstance(hierarchical_structure, HierarchicDataset):
            self.hierarchy_information = hierarchical_structure.to_hierarchy_information()
        elif isinstance(hierarchical_structure, HierarchyInformation):
            self.hierarchy_information = hierarchical_structure
        else:
            raise TypeError("hierarchical_structure must be HierarchicDataset or HierarchyInformation")
        
        # Initialize CLIP base head
        self.base = ClipBaseModel(clip_model_name, output_type=output_type, enable_checkpointing=enable_checkpointing)
        
        # Lazy initialization of level-specific classifier heads
        self._level_classifiers = {}
        self._num_head_dims = num_head_dims
        self._num_hidden_dims = num_hidden_dims
        self._heads = heads
        self._depth = depth
        self._enable_checkpointing = enable_checkpointing
        self._compile_args = None
        self._level_cluster_index_maps = {}
        
        # Initialize criterion and optimizer
        self.criterion = KLDivLossWithSoftmax()
        
        # Collect parameters from base (will add classifier params lazily)
        self.optimizer = bnb.optim.AdamW8bit([
            {'params': self.base.parameters(), 'lr': lr * 0.05}  # Lowest LR for pretrained CLIP
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
            'hierarchical_structure': self.hierarchy_information.clone(),
            'num_head_dims': num_head_dims,
            'num_hidden_dims': num_hidden_dims,
            'heads': heads,
            'depth': depth,
            'clip_model_name': clip_model_name,
            'output_type': output_type,
        }
    
    def _create_classifier_for_level(self, level_path: Tuple) -> nn.Module:
        """Lazily create a classifier head for a given hierarchy level."""
        # Get number of classes at this level
        if self.hierarchy_information.is_leaf(level_path):
            # Leaf level
            num_classes = self.hierarchy_information.get_leaf_size(level_path)
        else:
            # Internal level
            children = self.hierarchy_information.get_children(level_path)
            num_classes = len(children)
        
        # Create the classifier head
        seq = CheckpointedSequential if self._enable_checkpointing else nn.Sequential
        model = seq(
            nn.LayerNorm(self.base.logits_dim),
            nn.Linear(self.base.logits_dim, self._num_head_dims) if self.base.logits_dim != self._num_head_dims else nn.Identity(),
            FeaturePerspective(self._num_head_dims, self._num_head_dims, num_heads=self._heads),
            SkipAttentionMLP(self._num_head_dims, self._num_hidden_dims, depth=self._depth),
            nn.Linear(self._num_hidden_dims, num_classes)
        )
        
        # Add to optimizer (separate param groups for geo processor vs classifier)
        geo_processor_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'feature_perspective' in name.lower():
                geo_processor_params.append(param)
            else:
                classifier_params.append(param)
        
        self.optimizer.add_param_group({'params': geo_processor_params, 'lr': self.get_current_lr() * 0.5})
        self.optimizer.add_param_group({'params': classifier_params, 'lr': self.get_current_lr()})
        
        # Move to device
        if self.device is not None:
            model = model.to(device=self.device, dtype=self.dtype)
        
        # Compile if needed
        if self._compile_args is not None:
            model = torch.compile(model, **self._compile_args)
            
        return model        

    def _get_classifier_for_level(self, level_path: Tuple) -> nn.Module:
        """Return (and lazily create) the classifier head for a hierarchy level."""
        if level_path not in self._level_classifiers:
            self._level_classifiers[level_path] = self._create_classifier_for_level(level_path)
        return self._level_classifiers[level_path]

    def _get_level_depth(self, level_path: Tuple) -> int:
        """Return the depth index (0-based) corresponding to the next decision level."""
        return sum(value is not None for value in level_path)

    def _get_cluster_index_map(self, level_idx: int, cluster_ids: torch.Tensor) -> Dict[int, int]:
        """Return mapping from cluster id to column index for a given level."""
        if level_idx not in self._level_cluster_index_maps:
            ids_list = cluster_ids.detach().cpu().tolist()
            self._level_cluster_index_maps[level_idx] = {int(cluster_id): idx for idx, cluster_id in enumerate(ids_list)}
        return self._level_cluster_index_maps[level_idx]

    def _build_target_distribution(
        self,
        level_idx: int,
        hierarchical_logits,
        sample_idx: int,
        children: List,
    ) -> torch.Tensor | None:
        """Create target distribution aligned with the classifier's child ordering."""
        if level_idx not in hierarchical_logits or not children:
            return None

        cluster_ids, level_probs = hierarchical_logits[level_idx]
        cluster_map = self._get_cluster_index_map(level_idx, cluster_ids)
        sample_probs = level_probs[sample_idx:sample_idx+1]
        target = level_probs.new_zeros((1, len(children)))

        has_values = False
        for child_pos, child in enumerate(children):
            child_id = self._coerce_child_scalar(child)
            if child_id is None:
                continue
            column_idx = cluster_map.get(child_id)
            if column_idx is None:
                continue
            target[:, child_pos] = sample_probs[:, column_idx]
            has_values = True

        if not has_values:
            return None

        total = target.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return target / total

    def _normalize_hierarchy_path(self, hierarchy_path) -> Tuple:
        """Convert various hierarchy path representations to a normalized tuple."""
        if isinstance(hierarchy_path, torch.Tensor):
            values = hierarchy_path.tolist()
        elif isinstance(hierarchy_path, (list, tuple)):
            values = list(hierarchy_path)
        else:
            values = [hierarchy_path]

        normalized = tuple(None if v in (None, "") else int(v) for v in values)
        if len(normalized) != self.hierarchy_information.key_length:
            raise ValueError(
                f"Hierarchy path length {len(normalized)} does not match expected {self.hierarchy_information.key_length}."
            )
        return normalized

    @staticmethod
    def _coerce_child_scalar(child_value):
        """Return an int scalar for child identifiers when possible."""
        if isinstance(child_value, tuple):
            return None
        if torch.is_tensor(child_value):
            child_value = child_value.item()
        if child_value in (None, ""):
            return None
        return int(child_value)

    def _apply_child_value(self, current_level: Tuple, child_value) -> Tuple:
        """Advance to the next hierarchy level using the provided child identifier."""
        if isinstance(child_value, tuple):
            return child_value

        scalar_value = self._coerce_child_scalar(child_value)
        if scalar_value is None:
            return current_level

        new_level = list(current_level)
        try:
            none_index = new_level.index(None)
        except ValueError:
            return current_level

        new_level[none_index] = scalar_value
        return tuple(new_level)

    def _predict_leaf_from_feature(self, sample_feature: torch.Tensor) -> Tuple[Tuple, torch.Tensor]:
        """Traverse the hierarchy using model predictions and return the leaf node and logits."""
        current_level = tuple([None] * self.hierarchy_information.key_length)

        while True:
            classifier = self._get_classifier_for_level(current_level)
            output = classifier(sample_feature)
            children = self.hierarchy_information.get_children(current_level)

            if self.hierarchy_information.is_leaf(current_level) or not children:
                return current_level, output

            child_idx = output.argmax(dim=-1).item()
            child_idx = max(0, min(child_idx, len(children) - 1))
            current_level = self._apply_child_value(current_level, children[child_idx])

    def _match_child_for_path(self, current_level: Tuple, hierarchy_path: Tuple) -> Tuple:
        """Return the child tuple that matches the provided hierarchy path."""
        children = self.hierarchy_information.get_children(current_level)
        if not children:
            return current_level

        normalized_path = self._normalize_hierarchy_path(hierarchy_path)

        try:
            next_index = current_level.index(None)
        except ValueError:
            return current_level

        target_value = normalized_path[next_index]
        if target_value in (None, ""):
            raise ValueError(f"Hierarchy path {normalized_path} is missing value for level {current_level}.")

        for child in children:
            if isinstance(child, tuple) and child == normalized_path:
                return child

        scalar_children = {
            self._coerce_child_scalar(child): child for child in children if not isinstance(child, tuple)
        }

        coerced_target = self._coerce_child_scalar(target_value)
        if coerced_target not in scalar_children:
            raise ValueError(
                f"No child of level {current_level} matches hierarchy path {normalized_path}."
            )

        return self._apply_child_value(current_level, coerced_target)
    
    def train_batch(self, batch, transforms=None, accumulation_steps=3):
        """
        Train on a batch with hierarchical logits.
        
        Args:
         batch: Tuple of (images, hierarchical_logits, indices, hierarchy_paths)
             - images: Tensor of shape (batch_size, C, H, W)
             - hierarchical_logits: Dict[level_idx, (cluster_ids, Tensor)] of per-level targets
                   - indices: Tensor of global indices (optional, currently unused but retained for bookkeeping)
                   - hierarchy_paths: Sequence of tuples describing each sample's hierarchy path
        """
        self.base.train()
        for model in self._level_classifiers.values():
            model.train()
        
        # Unpack batch (indices are global dataset indices; hierarchy_paths derived from collate)
        images, hierarchical_logits, indices, hierarchy_paths = batch
        assert isinstance(indices, torch.Tensor), "Indices must be provided as a tensor"
        assert hierarchy_paths is not None, "Hierarchy paths must be provided for hierarchical loss"
        assert len(hierarchy_paths) == images.size(0), "Hierarchy path count must match batch size"
        
        # Apply transforms on-the-fly
        if transforms:
            images = transforms(images)
        
        # Split batch for accumulation
        step_size = max(1, images.size(0) // accumulation_steps)
        used_param_map = {}
        
        accumulated_loss = 0
        for i in range(accumulation_steps):
            start_idx = i * step_size
            if start_idx >= images.size(0):
                break
            end_idx = (i + 1) * step_size if i < accumulation_steps - 1 else images.size(0)
            
            sub_images = images[start_idx:end_idx]
            sub_paths = hierarchy_paths[start_idx:end_idx]
            sub_logits = {
                level: (ids, probs[start_idx:end_idx])
                for level, (ids, probs) in hierarchical_logits.items()
            }
            if sub_images.numel() == 0:
                continue
            
            # Get CLIP features once for all samples
            clip_features = self.base(sub_images)
            
            # Traverse hierarchy and compute loss at each level
            loss, used_params = self._compute_hierarchical_loss(clip_features, sub_logits, sub_paths)
            loss = loss / accumulation_steps
            
            if loss.isnan().any():
                warnings.warn(f"NaN loss encountered in batch {i+1}/{accumulation_steps}. Skipping.")
                continue
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            for param in used_params:
                used_param_map[id(param)] = param
        
        # Update weights after accumulation - only params that were used
        used_params = list(used_param_map.values()) if used_param_map else []
        all_params = list(self.base.parameters()) + used_params
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.total_batches_trained += 1
        return accumulated_loss
    
    def _compute_hierarchical_loss(self, clip_features: torch.Tensor, hierarchical_logits, hierarchy_paths) -> tuple[torch.Tensor, list]:
        """
        Compute loss by traversing down the hierarchy for each sample.
        Returns: (loss, list of parameters used in the forward pass)
        """
        total_loss = clip_features.sum() * 0
        batch_size = clip_features.size(0)
        used_classifiers = set()
        assert hierarchy_paths is not None, "Hierarchy paths required to compute hierarchical loss"
        assert len(hierarchy_paths) == batch_size, "Hierarchy path count must match batch size"
        
        # For each sample, traverse its hierarchy path
        for sample_idx in range(batch_size):
            hierarchy_path = self._normalize_hierarchy_path(hierarchy_paths[sample_idx])
            sample_feature = clip_features[sample_idx:sample_idx+1]
            current_level = tuple([None] * self.hierarchy_information.key_length)
            
            for depth in range(self.hierarchy_information.key_length):
                children = self.hierarchy_information.get_children(current_level)
                if not children:
                    break

                classifier = self._get_classifier_for_level(current_level)
                used_classifiers.add(id(classifier))
                output = classifier(sample_feature)

                target = self._build_target_distribution(depth, hierarchical_logits, sample_idx, children)
                if target is not None:
                    total_loss = total_loss + self.criterion(output, target)

                current_level = self._match_child_for_path(current_level, hierarchy_path)
        
        used_params = []
        for classifier in self._level_classifiers.values():
            if id(classifier) in used_classifiers:
                used_params.extend(list(classifier.parameters()))
        
        return total_loss / batch_size, used_params
    
    def update_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

    def evaluate(self, data_loader, transforms=None):
        """
        Evaluate the model on the validation set.
        Returns: (average_loss, average_hit_rate)
        """
        self.base.eval()
        for model in self._level_classifiers.values():
            model.eval()
        
        total_loss = 0.0
        total_hits = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for images, hierarchical_logits, hierarchy_paths, _indices in data_loader:
                if transforms:
                    images = transforms(images)
                
                images = images.to(self.device, dtype=self.dtype)
                hierarchy_paths = [self._normalize_hierarchy_path(path) for path in hierarchy_paths]
                hierarchical_logits = {
                    level: (
                        ids.to(self.device, dtype=torch.int64),
                        logits.to(self.device, dtype=self.dtype)
                    )
                    for level, (ids, logits) in hierarchical_logits.items()
                }
                
                # Get CLIP features
                clip_features = self.base(images)
                
                # Compute hierarchical loss
                loss, _ = self._compute_hierarchical_loss(clip_features, hierarchical_logits, hierarchy_paths)
                
                if loss.isnan().any():
                    warnings.warn("NaN loss encountered during evaluation. Skipping.")
                    continue
                
                # Predict paths and compare to targets using shared traversal helper
                batch_size = images.size(0)
                for sample_idx in range(batch_size):
                    sample_feature = clip_features[sample_idx:sample_idx+1]
                    predicted_leaf, _ = self._predict_leaf_from_feature(sample_feature)
                    total_hits += int(predicted_leaf == hierarchy_paths[sample_idx])
                
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        return total_loss / num_samples, total_hits / num_samples
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Return predicted logits for a single image, traversing the hierarchy.
        Returns a merged tensor where:
        - Predicted logits are placed at their correct global indices
        - Unpredicted logits are zero
        """
        self.base.eval()
        for model in self._level_classifiers.values():
            model.eval()
        
        if not self.hierarchy_information.leaf_offsets:
            raise RuntimeError("Hierarchy information is missing leaf_offsets; cannot map logits to global indices.")

        with torch.no_grad():
            clip_features = self.base(image)
            total_size = sum(self.hierarchy_information.leaf_counts.values())
            merged_logits = torch.zeros(image.size(0), total_size, device=image.device, dtype=image.dtype)

            batch_size = image.size(0)
            for sample_idx in range(batch_size):
                sample_feature = clip_features[sample_idx:sample_idx+1]
                leaf_level, leaf_output = self._predict_leaf_from_feature(sample_feature)

                if not self.hierarchy_information.is_leaf(leaf_level):
                    continue

                start, end = self.hierarchy_information.get_leaf_index_range(leaf_level)
                merged_logits[sample_idx, start:end] = leaf_output[0]

            return merged_logits
        
    def predict_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted probabilities for a single image."""
        logits = self(image)
        return torch.softmax(logits, dim=-1)
        
    def get_current_lr(self):
        """Return the current learning rate"""
        return max(group['lr'] for group in self.optimizer.param_groups)
    
    def get_model_size(self):
        """Returns the ON-DEVICE size of the model in bytes."""
        total_size = 0
        
        # Base model
        for param in self.base.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in self.base.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        # All classifier heads
        for classifier in self._level_classifiers.values():
            for param in classifier.parameters():
                total_size += param.numel() * param.element_size()
            for buffer in classifier.buffers():
                total_size += buffer.numel() * buffer.element_size()
        
        return total_size
    
    def send_to_device(self, device, dtype=None):
        """Sends the current model to the specified device and dtype."""
        if dtype is None:
            dtype = self.dtype
        
        # Helper to move parameters
        def move_params(module):
            for param in module.parameters():
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device=device, dtype=dtype)
            
            for buffer in module.buffers():
                if buffer.dtype in [torch.int32, torch.int64, torch.long]:
                    buffer.data = buffer.data.to(device=device)
                else:
                    buffer.data = buffer.data.to(device=device, dtype=dtype)
        
        # Move base model
        move_params(self.base)
        
        # Move all classifier heads
        for classifier in self._level_classifiers.values():
            move_params(classifier)
        
        self.criterion = self.criterion.to(device)
        self.device = device
        self.dtype = dtype
        
    def compile(self, **kwargs):
        """Compiles all classifier heads for optimized inference."""
        self._compile_args = kwargs
        
        # Compile base
        self.base = torch.compile(self.base, **kwargs)
        
        # Compile each classifier head
        compiled_classifiers = {}
        for level, classifier in self._level_classifiers.items():
            compiled_classifiers[level] = torch.compile(classifier, **kwargs)
        self._level_classifiers = compiled_classifiers
    
    def save(self, filepath):
        """Save model state to file."""
        # Extract state dicts from all classifiers
        classifier_state_dicts = {}
        for level, classifier in self._level_classifiers.items():
            if hasattr(classifier, '_orig_mod'):
                classifier_state_dicts[level] = classifier._orig_mod.state_dict()
            else:
                classifier_state_dicts[level] = classifier.state_dict()
        
        torch.save({
            'base_state_dict': self.base.state_dict(),
            'classifier_state_dicts': classifier_state_dicts,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'init_params': self.init_params,
            'total_batches_trained': self.total_batches_trained
        }, filepath)
    
    @staticmethod
    def load(filepath):
        """Load model state from file."""
        # Register hierarchy classes as safe for unpickling
        torch.serialization.add_safe_globals([HierarchicDataset, HierarchyInformation])
        
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model with saved init params
        model = HierarchicGeoClassifier(**checkpoint['init_params'])
        
        # Load base state
        model.base.load_state_dict(checkpoint['base_state_dict'])
        
        # Load classifier states
        for level, state_dict in checkpoint['classifier_state_dicts'].items():
            # Trigger lazy creation
            classifier = model._get_classifier_for_level(level)
            classifier.load_state_dict(state_dict)
        
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.total_batches_trained = checkpoint['total_batches_trained']
        
        return model