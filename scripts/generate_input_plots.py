"""
Generate input data plots and summary tables from EpiForecaster config.

This script loads a training config, creates a dataloader, and generates:
- Cases + mobility window figure
- LOCF biomarker figure with ribbons
- Biomarker sparsity analysis figure
- Summary CSV tables
"""

import argparse
import logging
import random
from pathlib import Path

from torch.utils.data import DataLoader

from models.configs import EpiForecasterConfig
from plotting.input_plots import (
    collect_case_window_samples,
    export_biomarker_sparsity_tables,
    export_summary_tables,
    make_biomarker_locf_figure,
    make_biomarker_sparsity_figure_all,
    make_cases_window_figure,
)
from utils.logging import setup_logging, suppress_zarr_warnings

suppress_zarr_warnings()
logger = logging.getLogger(__name__)


def generate_input_plots(
    config_path: str,
    output_dir: str | Path | None = None,
    num_samples: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    generate_plots: bool = True,
    generate_tables: bool = True,
) -> tuple[list, dict[str, Path]]:
    """Generate input visualization plots and summary tables from a training config.

    Args:
        config_path: Path to training config YAML
        output_dir: Output directory for plots and tables
        num_samples: Number of samples to plot
        shuffle: Whether to shuffle samples
        seed: Random seed
        generate_plots: Whether to generate plots
        generate_tables: Whether to generate summary tables

    Returns:
        Tuple of (samples, exported_files) where exported_files maps table name to path
    """
    setup_logging(logging.INFO)

    # Load config
    logger.info(f"Loading config from: {config_path}")
    config = EpiForecasterConfig.from_file(config_path)

    # Get all node IDs from dataset
    from data.epi_dataset import EpiDataset
    import xarray as xr

    zarr_path = Path(config.data.dataset_path).resolve()
    dataset = xr.open_zarr(zarr_path)
    from data.preprocess.config import REGION_COORD
    all_nodes = list(range(dataset[REGION_COORD].size))
    dataset.close()

    # Split nodes (train/val/test) - use a simple split for visualization
    rng = random.Random(seed)
    rng.shuffle(all_nodes)

    n_val = int(len(all_nodes) * config.training.val_split)
    n_test = int(len(all_nodes) * config.training.test_split)

    val_nodes = all_nodes[:n_val]
    train_nodes = all_nodes[n_val + n_test :]

    logger.info(f"Total nodes: {len(all_nodes)}")
    logger.info(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}")

    # Create dataset with train nodes for fitting preprocessors
    epi_dataset = EpiDataset(
        config=config,
        target_nodes=train_nodes,
        context_nodes=train_nodes + val_nodes,
    )

    # Create dataloader
    loader = DataLoader(
        epi_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Collect samples with LOCF biomarkers
    logger.info(f"Collecting {num_samples} samples (LOCF mode with ribbon)...")
    samples = collect_case_window_samples(
        loader=loader,
        n=num_samples,
        cases_feature_idx=0,
        biomarker_feature_idx=None,
        include_biomarkers=config.model.type.biomarkers and generate_plots,
        include_biomarkers_locf=config.model.type.biomarkers and generate_plots,
        include_mobility=config.model.type.mobility and generate_plots,
        shuffle=shuffle,
        seed=seed,
    )

    # Set output directory
    if output_dir is None:
        output_dir = f"outputs/input_viz_{Path(config_path).stem}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    history_length = config.model.history_length

    # For biomarker sparsity, we need the full biomarker data array
    biomarker_da = None
    if config.model.type.biomarkers:
        logger.info("Extracting full biomarker data for sparsity analysis...")
        full_dataset = xr.open_zarr(zarr_path)
        if "edar_biomarker" in full_dataset:
            biomarker_da = full_dataset.edar_biomarker
        full_dataset.close()

    if generate_plots:
        logger.info("Generating plots...")

        # Figure 1: Cases + mobility window
        if config.model.type.mobility or samples:
            logger.info("  - Creating cases window figure...")
            fig_cases = make_cases_window_figure(samples, history_length)
            if fig_cases is not None:
                cases_path = output_dir / "cases_window.png"
                fig_cases.savefig(cases_path, dpi=150, bbox_inches="tight")
                logger.info(f"    Saved: {cases_path}")

        # Figure 2: LOCF biomarkers with ribbons (always use ribbon)
        if config.model.type.biomarkers:
            logger.info("  - Creating LOCF biomarker figure with ribbons...")
            fig_locf = make_biomarker_locf_figure(
                samples, history_length, age_visualization="ribbon"
            )
            if fig_locf is not None:
                locf_path = output_dir / "biomarker_locf.png"
                fig_locf.savefig(locf_path, dpi=150, bbox_inches="tight")
                logger.info(f"    Saved: {locf_path}")

        # Figure 3: Biomarker sparsity analysis (all regions)
        if config.model.type.biomarkers and biomarker_da is not None:
            logger.info("  - Creating biomarker sparsity figure (all regions)...")
            fig_sparsity = make_biomarker_sparsity_figure_all(biomarker_da, history_length)
            if fig_sparsity is not None:
                sparsity_path = output_dir / "biomarker_sparsity.png"
                fig_sparsity.savefig(sparsity_path, dpi=200, bbox_inches="tight")
                logger.info(f"    Saved: {sparsity_path}")

    # Generate summary tables
    exported_tables = {}
    if generate_tables:
        logger.info("Generating summary tables...")

        # Biomarker tables (all regions)
        biomarker_tables = {}
        if config.model.type.biomarkers and biomarker_da is not None:
            logger.info("  - Generating biomarker sparsity tables (all regions)...")
            biomarker_tables = export_biomarker_sparsity_tables(biomarker_da, output_dir)
            exported_tables.update(biomarker_tables)
            for path in biomarker_tables.values():
                logger.info(f"    Saved: {path}")

        # Sample-level summary tables (skip biomarker tables since we exported all-region versions)
        sample_tables = export_summary_tables(
            samples, output_dir, skip_biomarker=(biomarker_da is not None)
        )
        exported_tables.update(sample_tables)
        for path in sample_tables.values():
            logger.info(f"  - Saved: {path}")

    logger.info(f"Done! Output directory: {output_dir}")
    return samples, exported_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate input data visualization plots and summary tables"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_epifor_mn5_full.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots and tables (default: outputs/input_viz_<config_name>)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle samples (default: shuffle)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots only, skip tables",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate tables only, skip plots",
    )

    args = parser.parse_args()

    # Determine what to generate
    generate_plots = not args.tables_only
    generate_tables = not args.plots_only

    generate_input_plots(
        config_path=args.config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        generate_plots=generate_plots,
        generate_tables=generate_tables,
    )
