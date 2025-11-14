# Copilot Instructions for GeolocateAI

## Big picture
- This repo trains CLIP-based geolocation models on multi-million-image datasets aggregated and clustered with HDBSCAN. Code lives under `modules/`, data prep scripts under `dev/data/`, and training entry points are `train_model_hierarchy.py` (multi-level classifier) and `train_model_nohierarchy.py` (flat classifier).
- Training relies on GPU acceleration, `torch` 2.6, `bitsandbytes` optimizers, and bfloat16. Expect long-running jobs and very large CSV inputs (`hierarchical_clustered_coords.csv`, etc.).

## Data workflow
- Raw datasets (MediaEval 2015/2016, Google Landmark, Flickr geotag) are staged under `dev/data/**`; follow that README for download + symlink instructions.
- Run `dev/data/combine_data.py` → `train_hdbscan.py` → `extract_hierarchy.py` to produce:
  - `hierarchical_clustered_coords.csv` (per-image clusters per level),
  - `hierarchical_cluster_centroids.csv` (level centroids + stats),
  - `hierarchical_structure.csv` (parent/child relationships + counts).
- Downstream scripts expect paths relative to `dev/data/`; keep filenames consistent or pass overrides via CLI flags.

## Training workflows
- Install deps via `poetry install` (Python ≥3.11 + CUDA). Use `poetry run python ...` to ensure consistent versions.
- Hierarchical training (`train_model_hierarchy.py`):
  - Loads CSVs with Polars, builds `HierarchyInformation` + `HierarchicDataset`, constructs `PerLevelSampler` (weighted by `create_sqrt_sampler`), and trains `HierarchicGeoClassifier` heads per hierarchy level.
  - Key flags: `--coords_file`, `--centroids_file`, `--hierarchy_file`, `--batch_size`, `--test_every`, `--save_every`, `--clip_model_name`, `--retrain`.
  - Checkpoints land in `models/` via `HierarchicGeoClassifier.save`; resuming uses `.load()` which only pickles hierarchy metadata, so after loading call `model.attach_dataset(train_dataset)` before evaluating or inferring.
- Flat training (`train_model_nohierarchy.py`):
  - Builds `ImageDataset` + `Geo*` classifiers. Choose base via `--base_model {clip,vt}` and whether to freeze CLIP via `--freeze_clip`.
  - Provided `train.sh` shows the expected batch sizes, compile flag, logging, and optional email notification.

## Architecture notes
- `modules/hierarchic_dataset.py` now defines `HierarchyInformation` (lightweight tree metadata) and `HierarchicDataset` (stores actual `ImageDataset` leaves + global indexing helpers). Clone hierarchy info when serializing, and re-use `attach_dataset()` whenever you recreate datasets.
- `HierarchicGeoClassifier` stacks CLIP base features with `FeaturePerspective`, `SkipAttentionMLP`, and per-level linear heads. Always route hierarchy traversal through `_match_child_for_path` and `_level_classifiers` so sampler + optimizer param groups stay in sync.
- `samplers.py` defines the canonical balancing strategies; reuse `create_sqrt_sampler` unless you have a reason to change global sampling weights.

## Conventions & tips
- CSV ingest is consistently done with Polars, casting cluster columns to `Int64` and filtering out noise clusters (`-1`); mirror that approach for new data utilities.
- Distance-based supervision uses haversine scoring normalized by cluster `mean_dist`/`std_dist`; keep tensors on the same device and dtype (usually `torch.bfloat16`).
- Many scripts silence extraneous logs (`TOKENIZERS_PARALLELISM`, `transformers` verbosity). Preserve these env vars when adding new entry points.
- When extending serialization, register new enums/classes with `torch.serialization.add_safe_globals` to keep checkpoints loadable under the weights-only default.
- Long dataloaders assume `num_workers=6`, `persistent_workers=True`, `prefetch_factor=3`, and pinned memory; align with these settings for new loaders unless you profile otherwise.
- Code should be minimal and modular; avoid monolithic scripts. Favor small utility functions in `modules/` over copy-pasting logic.
- Comments should only clarify non-obvious decisions, not restate what code does. Use docstrings only where args or method name does not encapsulate purpose.
- Mimic source style wherever possible.
- Ignore any line length requirement. Readability and clarity trump strict adherence to 80/100 character limits.
- Helper methods should be used in multiple places, dont make helpers that are only used once.

Keep this file short and pragmatic—update it whenever workflows or directory layouts change so AI agents can ramp up immediately.
