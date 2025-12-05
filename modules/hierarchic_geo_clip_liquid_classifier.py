import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Any
from collections import defaultdict
import warnings

from modules.base_classifier import AbstractGeoClassifier
from modules.model_factory import ModelFactory, ClipBackboneFactory, SkipAttentionMLPFactory
from modules.kldivlosssoftmax import KLDivLossWithSoftmax
from modules.schedulers import SmoothReduceLROnPlateau
from modules.hierarchic_dataset import HierarchicDataset, HierarchyInformation

torch.serialization.add_safe_globals([HierarchyInformation])

class HierarchicGeoClassifier(AbstractGeoClassifier):
    def __init__(self, 
                 hierarchical_structure: HierarchyInformation,
                 backbone_factory: ModelFactory = None,
                 head_factory: ModelFactory = None,
                 lr=0.001,
                 device=None, 
                 dtype=torch.float32,
                 **factory_kwargs):
        super().__init__(device, dtype)
        
        if hierarchical_structure is None:
            raise ValueError("hierarchical_structure must be provided")

        if isinstance(hierarchical_structure, HierarchicDataset):
            self.hierarchy_information = hierarchical_structure.to_hierarchy_information()
        elif isinstance(hierarchical_structure, HierarchyInformation):
            self.hierarchy_information = hierarchical_structure
        else:
            raise TypeError("hierarchical_structure must be HierarchicDataset or HierarchyInformation")
        
        # Initialize factories
        if backbone_factory is None:
            # Extract backbone specific kwargs if possible, or pass all
            # For simplicity, we assume factory_kwargs contains what's needed for ClipBackboneFactory
            # We need to filter or just pass relevant ones. 
            # ClipBackboneFactory takes: clip_model_name, output_type, enable_checkpointing
            self.backbone_factory = ClipBackboneFactory(
                clip_model_name=factory_kwargs.get('clip_model_name', "openai/clip-vit-large-patch14"),
                enable_checkpointing=factory_kwargs.get('enable_checkpointing', False)
            )
        else:
            self.backbone_factory = backbone_factory

        # Initialize base model FIRST so we get actual dimensions
        self.base = self.backbone_factory.create_model(freeze=True)
        
        # Get actual output dim from the created model (not heuristic)
        actual_backbone_dim = self.base.logits_dim
        
        if head_factory is None:
            # SkipAttentionMLPFactory takes: input_dim, output_dim, num_hidden_dims, heads, depth
            self.head_factory = SkipAttentionMLPFactory(
                input_dim=actual_backbone_dim,
                output_dim=0, # Will be set per level
                num_hidden_dims=factory_kwargs.get('num_hidden_dims', 2048),
                heads=factory_kwargs.get('heads', 8),
                depth=factory_kwargs.get('depth', 5)
            )
        else:
            self.head_factory = head_factory
            
        # Lazy initialization of level-specific classifier heads
        self._level_classifiers = {}
        self._level_cluster_index_maps = {}
        self.leaf_datapoint_counts = defaultdict(int)
        self._compile_args = None
        
        # Initialize criterion
        self.criterion = KLDivLossWithSoftmax()
        
        # Initialize optimizer with base parameters
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        
        self.lr = lr
        if base_params:
            # Use factory to get groups if possible, but here we just use manual or factory
            groups = self.backbone_factory.get_optimizer_param_groups(self.base, lr)
            self.optimizer = optim.AdamW(groups, lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = optim.AdamW([{'params': [], 'lr': lr}], lr=lr, weight_decay=1e-4)
            
        self.scheduler = SmoothReduceLROnPlateau(
            self.optimizer,
            smoothing_window=10,
            historical_window=100,
            reduction_threshold=0.95,
            cooldown=20,
            factor=0.8,
            min_lr=1e-5,
        )
        
        # Save init params for reconstruction
        self.init_params = {
            'hierarchical_structure': self.hierarchy_information.clone(),
            'lr': lr,
            **factory_kwargs,
        }
        
        if device:
            self.send_to_device(device, dtype)

    def _create_classifier_for_level(self, level_path: Tuple) -> nn.Module:
        """Lazily create a classifier head for a given hierarchy level using the factory."""
        # Get number of classes at this level
        if self.hierarchy_information.is_leaf(level_path):
            num_classes = self.hierarchy_information.get_leaf_size(level_path)
        else:
            children = self.hierarchy_information.get_children(level_path)
            num_classes = len(children)
        
        # Use factory to create head
        model = self.head_factory.create_model(output_dim=num_classes)
        
        # Move to device BEFORE adding to optimizer (optimizer needs correct dtype)
        if self.device is not None:
            model = model.to(device=self.device, dtype=self.dtype)
        
        # Add parameters to optimizer AFTER dtype conversion
        param_groups = self.head_factory.get_optimizer_param_groups(model, self.lr)
        for group in param_groups:
            self.optimizer.add_param_group(group)
        
        # Compile if needed
        if self._compile_args is not None:
            model = torch.compile(model, **self._compile_args)
            
        return model

    def _get_classifier_for_level(self, level_path: Tuple) -> nn.Module:
        if level_path not in self._level_classifiers:
            self._level_classifiers[level_path] = self._create_classifier_for_level(level_path)
        return self._level_classifiers[level_path]

    def train_batch(self, batch, transforms=None) -> float:
        self.base.train()
        for model in self._level_classifiers.values():
            model.train()
        
        images, hierarchical_logits, _, hierarchy_paths = batch
        
        if transforms:
            images = transforms(images)
        
        for leaf_path in hierarchy_paths:
            self.leaf_datapoint_counts[tuple(leaf_path)] += 1
        
        clip_features = self.base(images)
        
        loss, used_params = self._compute_hierarchical_loss(clip_features, hierarchical_logits, hierarchy_paths)
        
        if loss.isnan().any():
            warnings.warn("NaN loss encountered. Skipping.")
            return 0.0
        
        loss.backward()
        
        # Clip gradients
        all_params = list(self.base.parameters()) + used_params
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.total_batches_trained += 1
        return loss.item()

    def _compute_hierarchical_loss(self, clip_features: torch.Tensor, hierarchical_logits, hierarchy_paths) -> Tuple[torch.Tensor, List]:
        total_loss = clip_features.sum() * 0
        batch_size = clip_features.size(0)
        used_classifiers = set()
        
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

    def _normalize_hierarchy_path(self, hierarchy_path) -> Tuple:
        if isinstance(hierarchy_path, torch.Tensor):
            values = hierarchy_path.tolist()
        elif isinstance(hierarchy_path, (list, tuple)):
            values = list(hierarchy_path)
        else:
            values = [hierarchy_path]
        normalized = tuple(None if v in (None, "") else int(v) for v in values)
        return normalized

    def _match_child_for_path(self, current_level: Tuple, hierarchy_path: Tuple) -> Tuple:
        children = self.hierarchy_information.get_children(current_level)
        if not children: return current_level
        
        normalized_path = hierarchy_path
        try:
            next_index = current_level.index(None)
        except ValueError:
            return current_level

        target_value = normalized_path[next_index]
        
        for child in children:
            if isinstance(child, tuple) and child == normalized_path:
                return child
            if not isinstance(child, tuple):
                scalar_child = int(child) if child not in (None, "") else None
                if scalar_child == target_value:
                    new_level = list(current_level)
                    new_level[next_index] = scalar_child
                    return tuple(new_level)
        
        raise ValueError(f"No child of {current_level} matches path {normalized_path}")

    def _build_target_distribution(self, level_idx, hierarchical_logits, sample_idx, children):
        if level_idx not in hierarchical_logits or not children: return None
        cluster_ids, level_probs = hierarchical_logits[level_idx]
        
        if level_idx not in self._level_cluster_index_maps:
             ids_list = cluster_ids.detach().cpu().tolist()
             self._level_cluster_index_maps[level_idx] = {int(cid): i for i, cid in enumerate(ids_list)}
        cluster_map = self._level_cluster_index_maps[level_idx]
        
        sample_probs = level_probs[sample_idx:sample_idx+1]
        target = level_probs.new_zeros((1, len(children)))
        
        has_values = False
        for i, child in enumerate(children):
            child_id = int(child) if not isinstance(child, tuple) and child not in (None, "") else None
            if child_id is not None and child_id in cluster_map:
                target[:, i] = sample_probs[:, cluster_map[child_id]]
                has_values = True
                
        if not has_values: return None
        return target / (target.sum(dim=-1, keepdim=True).clamp_min(1e-9))

    def evaluate(self, data_loader, transforms=None) -> Tuple[float, float]:
        self.base.eval()
        for model in self._level_classifiers.values(): model.eval()
        
        total_loss = 0.0
        total_hits = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                images, hierarchical_logits, _, hierarchy_paths = batch
                if transforms: images = transforms(images)
                images = images.to(self.device, dtype=self.dtype)
                
                hierarchical_logits = {
                    l: (ids.to(self.device), probs.to(self.device, dtype=self.dtype))
                    for l, (ids, probs) in hierarchical_logits.items()
                }
                
                clip_features = self.base(images)
                loss, _ = self._compute_hierarchical_loss(clip_features, hierarchical_logits, hierarchy_paths)
                
                if not loss.isnan().any():
                    total_loss += loss.item() * images.size(0)
                
                for i in range(images.size(0)):
                    feat = clip_features[i:i+1]
                    pred_leaf, _ = self._predict_leaf_from_feature(feat)
                    norm_target = self._normalize_hierarchy_path(hierarchy_paths[i])
                    if pred_leaf == norm_target:
                        total_hits += 1
                
                num_samples += images.size(0)
                
        return total_loss / max(1, num_samples), total_hits / max(1, num_samples)

    def _predict_leaf_from_feature(self, sample_feature):
        current_level = tuple([None] * self.hierarchy_information.key_length)
        while True:
            classifier = self._get_classifier_for_level(current_level)
            output = classifier(sample_feature)
            children = self.hierarchy_information.get_children(current_level)
            
            if self.hierarchy_information.is_leaf(current_level) or not children:
                return current_level, output
            
            child_idx = output.argmax(dim=-1).item()
            child_idx = max(0, min(child_idx, len(children)-1))
            
            child = children[child_idx]
            if isinstance(child, tuple):
                current_level = child
            else:
                scalar = int(child) if child not in (None, "") else None
                try:
                    idx = current_level.index(None)
                    new_lvl = list(current_level)
                    new_lvl[idx] = scalar
                    current_level = tuple(new_lvl)
                except ValueError:
                    return current_level, output

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        for model in self._level_classifiers.values(): model.eval()
        
        if not self.hierarchy_information.leaf_offsets:
            raise RuntimeError("Hierarchy information is missing leaf_offsets")

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

    def _get_additional_save_state(self) -> Dict[str, Any]:
        classifier_states = {}
        for level, model in self._level_classifiers.items():
            classifier_states[level] = model.state_dict()
        
        return {
            'classifier_state_dicts': classifier_states,
            'leaf_datapoint_counts': dict(self.leaf_datapoint_counts),
            'base_state_dict': self.base.state_dict() if self.base else None
        }

    def _load_additional_state(self, checkpoint: Dict[str, Any]):
        if 'classifier_state_dicts' in checkpoint:
            for level, state in checkpoint['classifier_state_dicts'].items():
                classifier = self._get_classifier_for_level(level)
                classifier.load_state_dict(state)
        
        if 'leaf_datapoint_counts' in checkpoint:
            self.leaf_datapoint_counts = defaultdict(int, checkpoint['leaf_datapoint_counts'])
            
        if 'base_state_dict' in checkpoint and self.base:
            self.base.load_state_dict(checkpoint['base_state_dict'])

    def compile(self, **kwargs):
        self._compile_args = kwargs
        self.base = torch.compile(self.base, **kwargs)
        for model in self._level_classifiers.values():
            model = torch.compile(model, **kwargs)

    def send_to_device(self, device, dtype=None):
        super().send_to_device(device, dtype)
        if self.base:
            self.base.to(device=device, dtype=dtype)
        for model in self._level_classifiers.values():
            model.to(device=device, dtype=dtype)
