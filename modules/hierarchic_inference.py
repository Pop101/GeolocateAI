import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from modules.hierarchic_dataset import HierarchyInformation, LevelPath
from modules.base_classifier import AbstractGeoClassifier
from modules.model_factory import ModelFactory

class HierarchicInference(AbstractGeoClassifier):
    """
    Handles hierarchy traversal, leaf prediction, and model management (load/unload).
    Agnostic to specific backbone or head architectures.
    """
    def __init__(self, 
                 hierarchical_structure: HierarchyInformation,
                 backbone_factory: Optional[ModelFactory] = None,
                 head_factory: Optional[ModelFactory] = None,
                 device=None, 
                 dtype=torch.float32,
                 **factory_kwargs):
        super().__init__(device, dtype)
        
        if hierarchical_structure is None:
            raise ValueError("hierarchical_structure must be provided")

        # Handle HierarchyInformation
        if hasattr(hierarchical_structure, 'to_hierarchy_information'):
             self.hierarchy_information = hierarchical_structure.to_hierarchy_information()
        else:
             self.hierarchy_information = hierarchical_structure

        self.backbone_factory = backbone_factory
        self.head_factory = head_factory
        self._level_classifiers = {}
        self._compile_args = None
        
        # Initialize base model if factory provided
        if self.backbone_factory:
            # We assume the factory knows how to create the model with available kwargs
            # or that the model is created with defaults.
            # Subclasses can override this if they need specific creation logic.
            self.base = self.backbone_factory.create_model(**factory_kwargs)
        else:
            self.base = None
        
        # Save init params for reconstruction
        self.init_params = {
            'hierarchical_structure': self.hierarchy_information.clone(),
            **factory_kwargs,
        }
        
        if device:
            self.send_to_device(device, dtype)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input. 
        If a backbone (base) is present, runs it. Otherwise returns x.
        Subclasses can override for more complex feature extraction.
        """
        if self.base is not None:
            return self.base(x)
        return x

    def _create_classifier_for_level(self, level_path: LevelPath) -> nn.Module:
        """Lazily create a classifier head for a given hierarchy level using the factory."""
        if self.head_factory is None:
            raise NotImplementedError("head_factory not provided, cannot create classifier.")

        if self.hierarchy_information.is_leaf(level_path):
            num_classes = self.hierarchy_information.get_leaf_size(level_path)
        else:
            children = self.hierarchy_information.get_children(level_path)
            num_classes = len(children)
        
        # We assume the head factory accepts output_dim
        model = self.head_factory.create_model(output_dim=num_classes)
        
        if self.device is not None:
            model = model.to(device=self.device, dtype=self.dtype)
            
        if self._compile_args is not None:
            model = torch.compile(model, **self._compile_args)
            
        return model

    def _get_classifier_for_level(self, level_path: LevelPath) -> nn.Module:
        if level_path not in self._level_classifiers:
            self._level_classifiers[level_path] = self._create_classifier_for_level(level_path)
        return self._level_classifiers[level_path]

    def predict_leaf_from_feature(self, sample_feature: torch.Tensor) -> Tuple[LevelPath, torch.Tensor]:
        current_level = self.hierarchy_information.root_level()
        while True:
            classifier = self._get_classifier_for_level(current_level)
            output = classifier(sample_feature)
            children = self.hierarchy_information.get_children(current_level)

            if self.hierarchy_information.is_leaf(current_level) or not children:
                return current_level, output

            child_idx = output.argmax(dim=-1).item()
            child_idx = max(0, min(child_idx, len(children) - 1))

            child = children[child_idx]
            if isinstance(child, tuple):
                current_level = LevelPath(child)
            else:
                scalar = int(child)
                try:
                    idx = tuple(current_level).index(None)
                    new_lvl = list(current_level)
                    new_lvl[idx] = scalar
                    current_level = LevelPath(tuple(new_lvl))
                except ValueError:
                    return current_level, output

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        if self.base:
            self.base.eval()
        for model in self._level_classifiers.values(): model.eval()
        
        if not self.hierarchy_information.leaf_offsets:
            raise RuntimeError("Hierarchy information is missing leaf_offsets")

        with torch.no_grad():
            features = self.extract_features(image)

            total_size = sum(self.hierarchy_information.leaf_counts.values())
            merged_logits = torch.zeros(features.size(0), total_size, device=features.device, dtype=features.dtype)

            batch_size = features.size(0)
            for sample_idx in range(batch_size):
                sample_feature = features[sample_idx:sample_idx+1]
                leaf_level, leaf_output = self.predict_leaf_from_feature(sample_feature)

                if not self.hierarchy_information.is_leaf(leaf_level):
                    continue

                start, end = self.hierarchy_information.get_leaf_index_range(leaf_level)
                merged_logits[sample_idx, start:end] = leaf_output[0]

            return merged_logits

    def _get_additional_save_state(self) -> Dict[str, object]:
        classifier_states = {}
        for level, model in self._level_classifiers.items():
            classifier_states[level] = model.state_dict()
        
        return {
            'classifier_state_dicts': classifier_states,
            'base_state_dict': self.base.state_dict() if self.base else None,
        }

    def _load_additional_state(self, checkpoint: Dict[str, object]):
        if 'classifier_state_dicts' in checkpoint:
            for level, state in checkpoint['classifier_state_dicts'].items():
                classifier = self._get_classifier_for_level(level)
                classifier.load_state_dict(state)
            
        if 'base_state_dict' in checkpoint and self.base:
            self.base.load_state_dict(checkpoint['base_state_dict'])

    def send_to_device(self, device, dtype=None):
        super().send_to_device(device, dtype)
        if self.base:
            self.base.to(device=device, dtype=dtype)
        for model in self._level_classifiers.values():
            model.to(device=device, dtype=dtype)
            
    def compile(self, **kwargs):
        self._compile_args = kwargs
        if self.base:
            self.base = torch.compile(self.base, **kwargs)
        for level, model in self._level_classifiers.items():
            compiled = torch.compile(model, **kwargs)
            self._level_classifiers[level] = compiled
