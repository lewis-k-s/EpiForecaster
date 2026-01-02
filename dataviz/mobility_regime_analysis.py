"""Standalone analysis of mobility regime shifts.

This script ingests the raw OD mobility cube stored in
``data/files/mobility.zarr`` and computes two complementary feature views per
snapshot: topology-oriented (scale-reduced) and intensity-oriented
(scale-sensitive). It then builds regime signatures, estimates distances between
adjacent time steps, and surfaces potential change points grouped into
topology, intensity, or mixed regimes.

Example usage
-------------
python dataviz/mobility_regime_analysis.py \
    --mobility-path data/files/mobility.zarr \
    --output-dir outputs/reports/mobility_regimes \
    --edge-threshold 5.0 --laplacian-components 5 \
    --combined-threshold 2.5 --topology-threshold 2.0 --intensity-threshold 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

logger = logging.getLogger("mobility_regime_analysis")

# Threshold defaults tuned from empirical distribution of smoothed distances.
# We set the combined threshold near the 80–85th percentile to focus on the
# largest joint spikes while ignoring seasonal oscillations.
COMBINED_THRESHOLD = 5.4
# Topology threshold matches the upper quartile of smoothed topology distances
# so only marked rewiring events trigger topology-only alerts.
TOPOLOGY_THRESHOLD = 3.0
# Intensity threshold slightly exceeds the 75th percentile to catch only major
# volume shocks (e.g., lockdowns or holiday surges) instead of daily noise.
INTENSITY_THRESHOLD = 4.3
# Minimum gap debounces detections so that once we flag a regime, we wait 10
# days before allowing another.
MIN_GAP_DAYS = 10


@dataclass
class RegimeAnalysisConfig:
    """Runtime configuration for the mobility regime analyzer."""

    mobility_path: Path
    output_dir: Path
    laplacian_components: int = 5
    edge_threshold: float = 0.0
    top_edge_fraction: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    smoothing_window: int = 7
    combined_threshold: float = COMBINED_THRESHOLD
    topology_threshold: float = TOPOLOGY_THRESHOLD
    intensity_threshold: float = INTENSITY_THRESHOLD
    min_gap: int = MIN_GAP_DAYS
    max_snapshots: int | None = None
    log_every: int = 25


class MobilityRegimeAnalyzer:
    """Computes topology/intensity features and detects regime shifts."""

    def __init__(self, config: RegimeAnalysisConfig):
        self.config = config
        self._prepare_output_dir()

    def _prepare_output_dir(self) -> None:
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        dataset = self._open_dataset()
        feature_df = self._compute_feature_table(dataset)
        signature_data = self._build_signatures(feature_df)
        distance_df = self._compute_distances(signature_data)
        dates = pd.DatetimeIndex(feature_df.index)
        events_df = self._detect_change_points(distance_df, dates)
        self._persist_outputs(feature_df, distance_df, events_df)
        self._generate_plots(distance_df, events_df)
        self._print_summary(distance_df, events_df)

    def _open_dataset(self) -> xr.Dataset:
        logger.info("Opening mobility dataset from %s", self.config.mobility_path)
        ds = xr.open_zarr(self.config.mobility_path, chunks={"date": 1})
        if "target" in ds.dims:
            ds = ds.rename({"target": "destination"})
        if "destination" not in ds.dims:
            raise ValueError("Mobility dataset must contain a 'destination' dimension")

        required_vars = {"mobility"}
        missing = required_vars - set(ds.data_vars)
        if missing:
            raise ValueError(f"Dataset missing required variables: {missing}")

        logger.info(
            "Loaded dataset with %d snapshots and %d regions",
            ds.dims["date"],
            ds.dims["origin"],
        )
        return ds

    def _compute_feature_table(self, ds: xr.Dataset) -> pd.DataFrame:
        mobility = ds["mobility"].transpose("date", "origin", "destination")
        dates = pd.to_datetime(mobility["date"].values)
        records: list[dict[str, Any]] = []

        max_snapshots = (
            min(self.config.max_snapshots, mobility.sizes["date"])
            if self.config.max_snapshots is not None
            else mobility.sizes["date"]
        )

        for idx in range(max_snapshots):
            matrix = mobility.isel(date=idx).values
            intensity_features = compute_intensity_features(
                matrix,
                top_edge_fraction=self.config.top_edge_fraction,
            )
            topology_features = compute_topology_features(
                matrix,
                edge_threshold=self.config.edge_threshold,
                laplacian_components=self.config.laplacian_components,
            )
            record = {"date": dates[idx]}
            record.update(intensity_features)
            record.update(topology_features)
            records.append(record)

            if (idx + 1) % self.config.log_every == 0 or idx + 1 == max_snapshots:
                logger.info("Processed %d/%d snapshots", idx + 1, max_snapshots)

        feature_df = pd.DataFrame(records).set_index("date").sort_index()
        feature_df = feature_df.fillna(0.0)
        return feature_df

    def _build_signatures(self, feature_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        topology_cols = [c for c in feature_df.columns if c.startswith("topo_")]
        intensity_cols = [c for c in feature_df.columns if c.startswith("int_")]

        topology_sig = (
            standardize_dataframe(feature_df[topology_cols])
            if topology_cols
            else pd.DataFrame(index=feature_df.index)
        )
        intensity_sig = (
            standardize_dataframe(feature_df[intensity_cols])
            if intensity_cols
            else pd.DataFrame(index=feature_df.index)
        )

        combined_values = []
        for idx in range(len(feature_df)):
            topo_vec = (
                topology_sig.iloc[idx].to_numpy(dtype=float)
                if not topology_sig.empty
                else np.array([])
            )
            int_vec = (
                intensity_sig.iloc[idx].to_numpy(dtype=float)
                if not intensity_sig.empty
                else np.array([])
            )
            combined_values.append(
                np.concatenate(
                    [
                        self.config.alpha * topo_vec,
                        self.config.beta * int_vec,
                    ]
                )
            )

        combined_sig = pd.DataFrame(
            combined_values,
            index=feature_df.index,
        )

        return {
            "topology": topology_sig,
            "intensity": intensity_sig,
            "combined": combined_sig,
        }

    def _compute_distances(self, signatures: dict[str, pd.DataFrame]) -> pd.DataFrame:
        results = {}
        for key, df in signatures.items():
            if df.empty:
                dist = np.zeros(len(signatures["combined"]))
            else:
                dist = pairwise_distances(df.to_numpy(dtype=float))
            smoothed = smooth_series(dist, window=self.config.smoothing_window)
            results[f"d_{key}"] = dist
            results[f"d_{key}_smoothed"] = smoothed

        distance_df = pd.DataFrame(results, index=signatures["combined"].index)
        return distance_df

    def _detect_change_points(
        self, distance_df: pd.DataFrame, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        events: list[dict[str, Any]] = []
        last_event_idx = -np.inf

        combined = distance_df["d_combined_smoothed"].to_numpy()
        topo = distance_df["d_topology_smoothed"].to_numpy()
        intensity = distance_df["d_intensity_smoothed"].to_numpy()

        for idx in range(1, len(dates)):
            if combined[idx] < self.config.combined_threshold:
                continue
            if idx - last_event_idx < self.config.min_gap:
                continue

            event_type = classify_regime(
                d_topology=topo[idx],
                d_intensity=intensity[idx],
                topo_threshold=self.config.topology_threshold,
                intensity_threshold=self.config.intensity_threshold,
            )

            events.append(
                {
                    "date": dates[idx],
                    "d_total": combined[idx],
                    "d_topology": topo[idx],
                    "d_intensity": intensity[idx],
                    "regime_type": event_type,
                }
            )
            last_event_idx = idx

        return pd.DataFrame(events)

    def _persist_outputs(
        self,
        feature_df: pd.DataFrame,
        distance_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> None:
        feature_path = self.config.output_dir / "mobility_regime_features.parquet"
        distance_path = self.config.output_dir / "mobility_regime_distances.parquet"
        events_path = self.config.output_dir / "mobility_regime_change_points.parquet"

        feature_df.to_parquet(feature_path)
        distance_df.to_parquet(distance_path)
        if not events_df.empty:
            events_df.to_parquet(events_path)
        config_payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self.config).items()
        }

        summary = {
            "feature_file": str(feature_path),
            "distance_file": str(distance_path),
            "events_file": str(events_path) if not events_df.empty else None,
            "config": config_payload,
            "event_counts": events_df["regime_type"].value_counts().to_dict()
            if not events_df.empty
            else {},
        }
        with (self.config.output_dir / "mobility_regime_summary.json").open("w") as fh:
            json.dump(summary, fh, default=_serialize_datetime, indent=2)

    def _generate_plots(
        self, distance_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> None:
        time_index = pd.DatetimeIndex(distance_df.index)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            time_index,
            distance_df["d_topology_smoothed"],
            label="Topology",
            color="#1f77b4",
        )
        ax.plot(
            time_index,
            distance_df["d_intensity_smoothed"],
            label="Intensity",
            color="#ff7f0e",
        )
        ax.plot(
            time_index,
            distance_df["d_combined_smoothed"],
            label="Combined",
            color="#2ca02c",
        )
        if not events_df.empty:
            ax.scatter(
                events_df["date"],
                events_df["d_total"],
                c="red",
                s=15,
                label="Change points",
                alpha=0.7,
            )
        ax.set_title("Smoothed Regime Distances Over Time")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.config.output_dir / "mobility_regime_distances.png", dpi=200)
        plt.close(fig)

        if not events_df.empty:
            counts = events_df["regime_type"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(counts.index, counts.values, color="#9467bd")
            ax.set_title("Regime Change Counts by Type")
            ax.set_ylabel("Count")
            ax.set_xlabel("Regime type")
            for idx, value in enumerate(counts.values):
                ax.text(idx, value + 0.5, str(value), ha="center")
            fig.tight_layout()
            fig.savefig(self.config.output_dir / "mobility_regime_counts.png", dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 6))
            palette = {
                "topology": "#1f77b4",
                "intensity": "#ff7f0e",
                "mixed": "#2ca02c",
                "combined": "#d62728",
            }
            colors = events_df["regime_type"].map(palette).fillna("#7f7f7f")
            ax.scatter(
                events_df["d_topology"],
                events_df["d_intensity"],
                c=colors,
                alpha=0.8,
            )
            ax.axvline(
                self.config.topology_threshold,
                color="#1f77b4",
                linestyle="--",
                alpha=0.6,
            )
            ax.axhline(
                self.config.intensity_threshold,
                color="#ff7f0e",
                linestyle="--",
                alpha=0.6,
            )
            ax.set_xlabel("Topology distance")
            ax.set_ylabel("Intensity distance")
            ax.set_title("Change Points by Topology vs Intensity Signal")
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=key.capitalize(),
                    markerfacecolor=color,
                    markersize=8,
                )
                for key, color in palette.items()
            ]
            ax.legend(handles=handles, title="Regime type", loc="upper left")
            fig.tight_layout()
            fig.savefig(self.config.output_dir / "mobility_regime_scatter.png", dpi=200)
            plt.close(fig)

    def _print_summary(
        self, distance_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> None:
        logger.info("Computed distance statistics:")
        logger.info(distance_df.describe())
        if events_df.empty:
            logger.info("No change points detected with the current thresholds.")
        else:
            logger.info("Detected %d change points", len(events_df))
            logger.info(events_df.head())


def compute_intensity_features(
    matrix: np.ndarray,
    top_edge_fraction: float = 0.1,
) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    matrix[matrix < 0] = 0.0
    total_volume = matrix.sum()
    out_strength = matrix.sum(axis=1)
    in_strength = matrix.sum(axis=0)
    edge_values = matrix[matrix > 0]

    features: dict[str, float] = {
        "int_total_volume": float(total_volume),
        "int_log_total_volume": float(np.log1p(total_volume)),
        "int_intra_fraction": float(np.trace(matrix) / total_volume)
        if total_volume > 0
        else 0.0,
    }

    features.update(summarize_vector(out_strength, prefix="int_out_strength"))
    features.update(summarize_vector(in_strength, prefix="int_in_strength"))

    if edge_values.size == 0:
        features.update(
            {
                "int_edge_mean": 0.0,
                "int_edge_std": 0.0,
                "int_top_edge_fraction": 0.0,
                "int_edge_gini": 0.0,
            }
        )
    else:
        edge_mean = edge_values.mean()
        top_k = max(int(np.ceil(edge_values.size * top_edge_fraction)), 1)
        sorted_edges = np.sort(edge_values)
        top_sum = sorted_edges[-top_k:].sum()
        features.update(
            {
                "int_edge_mean": float(edge_mean),
                "int_edge_std": float(edge_values.std()),
                "int_top_edge_fraction": float(top_sum / total_volume)
                if total_volume > 0
                else 0.0,
                "int_edge_gini": float(gini_coefficient(edge_values)),
            }
        )

    return features


def compute_topology_features(
    matrix: np.ndarray,
    edge_threshold: float = 0.0,
    laplacian_components: int = 5,
) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    matrix[matrix < 0] = 0.0
    if edge_threshold > 0:
        matrix = np.where(matrix >= edge_threshold, matrix, 0.0)
    symmetric = 0.5 * (matrix + matrix.T)
    sparse_graph = csr_matrix(symmetric)

    degree = np.array(sparse_graph.sum(axis=1)).ravel()
    nz_edges = sparse_graph.count_nonzero()
    total_edges = symmetric.shape[0] * symmetric.shape[1]
    features: dict[str, float] = {
        "topo_edge_density": float(nz_edges / total_edges),
        "topo_mean_degree": float(degree.mean()),
        "topo_degree_std": float(degree.std()),
        "topo_degree_max": float(degree.max()) if degree.size else 0.0,
    }

    component_count, labels = connected_components(
        sparse_graph, directed=False, connection="weak"
    )
    features["topo_component_count"] = float(component_count)
    if labels.size:
        counts = np.bincount(labels)
        features["topo_largest_component_ratio"] = float(counts.max() / counts.sum())
    else:
        features["topo_largest_component_ratio"] = 0.0

    eigvals = laplacian_eigenvalues(
        sparse_graph,
        degree,
        k=laplacian_components,
    )
    for idx, eig in enumerate(eigvals):
        features[f"topo_lap_eig_{idx}"] = float(eig)

    return features


def summarize_vector(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
        }

    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p25": float(quantiles[0]),
        f"{prefix}_p50": float(quantiles[1]),
        f"{prefix}_p75": float(quantiles[2]),
        f"{prefix}_p90": float(quantiles[3]),
    }


def gini_coefficient(values: np.ndarray) -> float:
    values = values[values >= 0]
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_vals)) / (n * total)
    return float(gini)


def laplacian_eigenvalues(graph: csr_matrix, degree: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0)
    n = graph.shape[0]
    if n <= 2:
        return np.zeros(k)

    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    d_inv = diags(inv_sqrt)
    laplacian = eye(n, format="csr") - d_inv @ graph @ d_inv

    k = min(k, max(n - 2, 1))
    try:
        eigvals = eigsh(
            laplacian,
            k=k,
            which="SM",
            return_eigenvectors=False,
        )
        eigvals = np.sort(np.real(eigvals))
    except Exception as exc:  # pragma: no cover - diagnostic output
        logger.warning("Failed to compute Laplacian eigenvalues: %s", exc)
        eigvals = np.full(k, np.nan)

    if eigvals.size < k:
        eigvals = np.pad(eigvals, (0, k - eigvals.size), constant_values=np.nan)

    return eigvals


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean()
    std = df.std(ddof=0).replace(0, 1.0)
    return (df - mean) / std


def pairwise_distances(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    distances = np.zeros(len(values))
    if len(values) == 1:
        return distances
    diffs = np.diff(values, axis=0)
    distances[1:] = np.linalg.norm(diffs, axis=1)
    return distances


def smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    ser = pd.Series(series)
    return ser.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def classify_regime(
    d_topology: float,
    d_intensity: float,
    topo_threshold: float,
    intensity_threshold: float,
) -> str:
    topo_signal = d_topology >= topo_threshold
    intensity_signal = d_intensity >= intensity_threshold
    if topo_signal and intensity_signal:
        return "mixed"
    if topo_signal:
        return "topology"
    if intensity_signal:
        return "intensity"
    return "combined"


def _serialize_datetime(value: Any) -> Any:
    if isinstance(value, pd.Timestamp | np.datetime64):
        return str(pd.Timestamp(value))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mobility regime shifts")
    parser.add_argument(
        "--mobility-path",
        type=Path,
        default=Path("data/files/mobility.zarr"),
        help="Path to the mobility Zarr store",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/mobility_regimes"),
        help="Directory to store analysis artifacts",
    )
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    parser.add_argument("--top-edge-fraction", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--combined-threshold", type=float, default=COMBINED_THRESHOLD)
    parser.add_argument("--topology-threshold", type=float, default=TOPOLOGY_THRESHOLD)
    parser.add_argument(
        "--intensity-threshold", type=float, default=INTENSITY_THRESHOLD
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=MIN_GAP_DAYS,
        help="Minimum gap between change points",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Optional limit on number of snapshots to process",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = RegimeAnalysisConfig(
        mobility_path=args.mobility_path,
        output_dir=args.output_dir,
        laplacian_components=args.laplacian_components,
        edge_threshold=args.edge_threshold,
        top_edge_fraction=args.top_edge_fraction,
        alpha=args.alpha,
        beta=args.beta,
        smoothing_window=args.smoothing_window,
        combined_threshold=args.combined_threshold,
        topology_threshold=args.topology_threshold,
        intensity_threshold=args.intensity_threshold,
        min_gap=args.min_gap,
        max_snapshots=args.max_snapshots,
        log_every=args.log_every,
    )

    analyzer = MobilityRegimeAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()
