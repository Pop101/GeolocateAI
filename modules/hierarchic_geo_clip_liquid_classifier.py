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
from modules.hierarchic_dataset import HierarchicDataset, HierarchyInformation, LeafPath


def _ensure_leaf_path(path_like) -> LeafPath:
    # LeafPath is a typing.NewType wrapper around tuple[int, ...] and is not a runtime type.
    # Treat any tuple as a leaf path (caller guarantees it's full-length).
    if isinstance(path_like, tuple):
        return LeafPath(tuple(int(v) for v in path_like))
    if isinstance(path_like, list):
        return LeafPath(tuple(int(v) for v in path_like))
    if torch.is_tensor(path_like):
        if path_like.dim() == 0:
            raise ValueError(f"Expected LeafPath with length >= 1, got scalar tensor: {path_like}")
        return LeafPath(tuple(int(v) for v in path_like.tolist()))
    raise TypeError(f"Unsupported hierarchy path type: {type(path_like)}")
from modules.hierarchic_inference import HierarchicInference

torch.serialization.add_safe_globals([HierarchyInformation])

class HierarchicGeoClassifier(HierarchicInference):
    def __init__(self, 
                 hierarchical_structure: HierarchyInformation,
                 backbone_factory: ModelFactory = None,
                 head_factory: ModelFactory = None,
                 lr=0.001,
                 device=None, 
                 dtype=torch.float32,
                 **factory_kwargs):
        
        # Initialize backbone factory if not provided
        if backbone_factory is None:
            backbone_factory = ClipBackboneFactory(
                clip_model_name=factory_kwargs.get('clip_model_name', "openai/clip-vit-large-patch14"),
                enable_checkpointing=factory_kwargs.get('enable_checkpointing', False)
            )
            
        # Ensure freeze=True is passed for the backbone creation if not specified
        if 'freeze' not in factory_kwargs:
            factory_kwargs['freeze'] = True

        # Initialize super class (creates self.base)
        super().__init__(hierarchical_structure, backbone_factory, head_factory=None, device=device, dtype=dtype, **factory_kwargs)
        
        # Initialize head factory using actual backbone dimensions
        if head_factory is None:
            actual_backbone_dim = self.base.logits_dim
            self.head_factory = SkipAttentionMLPFactory(
                input_dim=actual_backbone_dim,
                output_dim=0, # Will be set per level
                num_hidden_dims=factory_kwargs.get('num_hidden_dims', 2048),
                heads=factory_kwargs.get('heads', 8),
                depth=factory_kwargs.get('depth', 5)
            )
        else:
            self.head_factory = head_factory
        
        # Initialize criterion
        self.criterion = KLDivLossWithSoftmax()
        
        # Initialize optimizer with base parameters
        base_params = [p for p in self.base.parameters() if p.requires_grad]
        
        self.lr = lr
        if base_params:
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
        
        self.leaf_datapoint_counts = defaultdict(int)
        self._level_cluster_index_maps = {}
        
        # Update init params
        self.init_params['lr'] = lr

    def _create_classifier_for_level(self, level_path: Tuple) -> nn.Module:
        """Lazily create a classifier head for a given hierarchy level using the factory."""
        model = super()._create_classifier_for_level(level_path)
        
        # Add parameters to optimizer
        param_groups = self.head_factory.get_optimizer_param_groups(model, self.lr)
        for group in param_groups:
            self.optimizer.add_param_group(group)
            
        return model

    def train_batch(self, batch, transforms=None) -> float:
        self.base.train()
        for model in self._level_classifiers.values():
            model.train()
        
        images, hierarchical_logits, _, hierarchy_paths = batch

        images, hierarchical_logits = self._move_batch_to_device(images, hierarchical_logits)
        
        if transforms:
            images = transforms(images)
        
        for leaf_path in hierarchy_paths:
            self.leaf_datapoint_counts[tuple(leaf_path)] += 1
        
        clip_features = self.extract_features(images)
        
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
            leaf_path = _ensure_leaf_path(hierarchy_paths[sample_idx])
            sample_feature = clip_features[sample_idx:sample_idx+1]
            current_level = self.hierarchy_information.root_level()
            
            for depth in range(self.hierarchy_information.key_length):
                children = self.hierarchy_information.get_children(current_level)
                if not children:
                    break

                classifier = self._get_classifier_for_level(current_level)
                used_classifiers.add(classifier)
                output = classifier(sample_feature)

                target = self._build_target_distribution(depth, hierarchical_logits, sample_idx, children)
                if target is not None:
                    total_loss = total_loss + self.criterion(output, target)

                current_level = self.hierarchy_information.advance_to_child(current_level, leaf_path)
        
        used_params = []
        for classifier in used_classifiers:
            used_params.extend(list(classifier.parameters()))
        
        return total_loss / batch_size, used_params

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
                images, hierarchical_logits = self._move_batch_to_device(images, hierarchical_logits)
                
                clip_features = self.extract_features(images)
                loss, _ = self._compute_hierarchical_loss(clip_features, hierarchical_logits, hierarchy_paths)
                
                if not loss.isnan().any():
                    total_loss += loss.item() * images.size(0)
                
                for i in range(images.size(0)):
                    feat = clip_features[i:i+1]
                    pred_leaf, _ = self.predict_leaf_from_feature(feat)
                    if tuple(pred_leaf) == tuple(hierarchy_paths[i]):
                        total_hits += 1
                
                num_samples += images.size(0)
                
        return total_loss / max(1, num_samples), total_hits / max(1, num_samples)

    def _get_additional_save_state(self) -> Dict[str, Any]:
        state = super()._get_additional_save_state()
        state['leaf_datapoint_counts'] = dict(self.leaf_datapoint_counts)
        return state

    def _load_additional_state(self, checkpoint: Dict[str, Any]):
        super()._load_additional_state(checkpoint)
        if 'leaf_datapoint_counts' in checkpoint:
            self.leaf_datapoint_counts = defaultdict(int, checkpoint['leaf_datapoint_counts'])
