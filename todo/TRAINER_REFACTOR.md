# Trainer and Utilities Refactor Plan

This document outlines a structural refactoring plan to break down monolithic files (such as `training/epiforecaster_trainer.py`, `utils/training_utils.py`, `utils/dtypes.py`, `utils/train_logging.py`, and `evaluation/epiforecaster_eval.py`) into more cohesive, focused modules.

## Goals

- Reduce the file size and complexity of the main trainer class.
- Separate data loading/preparation concerns from model training loops.
- Isolate device and precision handling from high-level data definitions.
- Separate pure logging logic from UI formatting and computational side-effects.
- Decompose the evaluation pipeline into logical components.

## Proposed Refactoring Architecture

### 1. `training/epiforecaster_trainer.py` (Currently Monolithic)

The `EpiForecasterTrainer` class has grown to handle too many responsibilities. We will extract the following specific factories and managers:

- [x] **`data/dataset_factory.py` (or `data/curriculum_setup.py`)**
  - **Responsibility**: Move the massive `__init__` block that handles temporal vs node-based splits, curriculum dataset instantiation, finding synthetic runs, and ensuring dimension consistency.
  - **Goal**: The trainer's `__init__` should just call `train_ds, val_ds, test_ds = build_datasets(config)` and focus purely on model training.
- [x] **`training/dataloader_factory.py`**
  - **Responsibility**: Extract `_create_data_loaders`, hardware-aware worker context selection (`_should_prestart_dataloader_workers`), and pin memory logic.
- [ ] **`training/optim_factory.py`**
  - **Responsibility**: Isolate `_create_optimizer` (handling of `AdamW`, `fused=True`, etc.) and `_create_scheduler`.
- [ ] **`training/gradnorm_manager.py` (or extend `training/gradnorm.py`)**
  - **Responsibility**: Move `_gradnorm_sidecar_update` and formatting code into a proper handler class that tracks its own state, cleaning up the main training loop.

### 2. `utils/training_utils.py` (Mixed Concerns)

Currently mixes generic tensor utilities with data pipeline-specific logic.

- [ ] **`data/mobility_utils.py` (or move into `data/epi_dataset.py` / `data/compiled_batch.py`)**
  - **Responsibility**: `inject_gpu_mobility` and `ensure_mobility_adj_dense_ready`. These strictly concern the handling of `MobBatch` and data pipeline structures.
- [x] **`data/epi_batch.py` (extend with collate functions)**
  - **Responsibility**: Move `collate_epiforecaster_batch`, `optimized_collate_graphs` from `data/epi_dataset.py`, and `mask_ablated_inputs` from `utils/training_utils.py`. Centralize all collation logic with the `EpiBatch` dataclass for better cohesion. Used by both trainer and eval.
- [x] **`utils/training_utils.py` (Generic Logic)**
  - **Responsibility**: Keep pure training logic like `drop_nowcast`, `get_effective_optimizer_step`, and `should_log_step`.

### 3. `utils/dtypes.py` (Storage vs Computation)

Mixes data preprocessing schemas with model execution types.

- [x] **`data/dtypes.py`**
  - **Responsibility**: Extract `STORAGE_DTYPES`, `NUMPY_STORAGE_DTYPES`, `get_storage_dtype`, and `is_storage_dtype`. Keep all references to raw data schema here.
- [x] **`utils/torch_utils.py` (or keep as `utils/dtypes.py` / rename to `utils/device.py`)**
  - **Responsibility**: Keep `MODEL_DTYPE_CPU`, `MODEL_DTYPE_GPU`, `AUTOCAST_DTYPE_*`. Keep safe epsilon getters (`get_dtype_safe_eps`, `get_model_eps`). Keep `sync_to_device`.

### 4. `utils/train_logging.py` (Mixed Operations & Formatting)

Mixes W&B payload logic, Console string formatting, and gradient mutations.

- [ ] **`utils/formatting.py` (or `utils/console.py`)**
  - **Responsibility**: Extract all string formatting functions like `format_train_progress_status`, `format_component_gradnorm_status`, `format_joint_loss_components_status`, `format_horizon_status_lines`.
- [ ] **`training/grad_utils.py`**
  - **Responsibility**: Move `compute_gradient_norms_and_clip` out of the logging module since it actually performs `clip_grad_norm_` which is a critical mutation step during training, not just logging.
- [x] **`utils/train_logging.py`**
  - **Responsibility**: Focus entirely on assembling structured dictionary payloads for Weights & Biases (e.g., `build_train_step_log_data`, `build_epoch_logging_bundle`).

### 5. `evaluation/epiforecaster_eval.py` (Overly Large - >2000 lines)

Combines loss definitions, dataloader construction, model loading, evaluation loops, and plotting generation.

- [ ] **`evaluation/losses.py`**
  - **Responsibility**: Extract `ForecastLoss`, `JointInferenceLoss`, `SMAPELoss`, `CompositeLoss`, and `get_loss_from_config`. This is core domain logic and takes up ~800 lines alone.
- [ ] **`evaluation/loaders.py`**
  - **Responsibility**: Extract `load_model_from_checkpoint`, `build_loader_from_config`, and `split_nodes`. This isolates the data-loading layer for inference.
- [ ] **`evaluation/selection.py`**
  - **Responsibility**: Extract node selection mechanisms (`select_nodes_by_loss`, `topk_target_nodes_by_mae`).
- [ ] **`evaluation/eval_loop.py`**
  - **Responsibility**: Keep `evaluate_loader` and `eval_checkpoint`. These are the core evaluation loop runners.

### 6. `utils/gradient_debug.py` (Solid)

- [x] **Status**: Good as is. This file is cohesive and well-structured, providing a clear interface (`GradientDebugger`) with zero overhead when disabled.
