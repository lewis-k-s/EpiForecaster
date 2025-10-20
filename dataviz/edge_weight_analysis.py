"""
Edge Weight Distribution Analysis for OD Network Data

Analyzes the distribution of edge weights in mobility OD networks to assess
whether there's sufficient separation for neural network models to learn
meaningful neighborhood weights.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

from constants import (
    LOCKDOWNS,
    RESTRICTION_LEVELS,
    STATE_OF_EMERGENCY,
)

logger = logging.getLogger(__name__)


class EdgeWeightAnalyzer:
    """Analyzes edge weight distributions in OD mobility networks."""

    def __init__(self, data_dir: str = "data/files/daily_dynpop_mitma/"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.lockdown_periods = LOCKDOWNS
        self._self_edge_cache = {}  # Cache self-edge mappings per dataset

    def load_all_data(self) -> Dict[str, xr.Dataset]:
        """Load all monthly OD data files."""
        datasets = {}
        nc_files = sorted(self.data_dir.glob("*.nc"))

        logger.info(f"Loading {len(nc_files)} OD data files...")

        for file_path in nc_files:
            # Extract date from filename
            filename = file_path.stem
            date_part = filename.split(".")[-1]  # e.g., "2020-02-01_2020-02-29"
            month_key = date_part.split("_")[0][:7]  # e.g., "2020-02"

            try:
                ds = xr.open_dataset(file_path, engine="h5netcdf")
                datasets[month_key] = ds
                logger.info(f"Loaded {month_key}: {ds.person_hours.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return datasets

    def _get_self_edge_mapping(self, ds: xr.Dataset) -> list[tuple[int, int]]:
        """Get mapping of self-edge indices for a dataset.

        Args:
            ds: xarray Dataset with home and destination coordinates

        Returns:
            List of (home_idx, dest_idx) tuples for self-edges (same region ID)
        """
        ds_id = id(ds)  # Use object ID as cache key

        if ds_id not in self._self_edge_cache:
            home_to_idx = {
                region_id: idx for idx, region_id in enumerate(ds.home.values)
            }
            dest_to_idx = {
                region_id: idx for idx, region_id in enumerate(ds.destination.values)
            }

            home_ids = set(ds.home.values)
            dest_ids = set(ds.destination.values)
            overlap = home_ids.intersection(dest_ids)

            self_edge_pairs = []
            for region_id in overlap:
                if region_id in home_to_idx and region_id in dest_to_idx:
                    home_idx = home_to_idx[region_id]
                    dest_idx = dest_to_idx[region_id]
                    self_edge_pairs.append((home_idx, dest_idx))

            self._self_edge_cache[ds_id] = self_edge_pairs
            logger.info(f"Cached {len(self_edge_pairs)} self-edge mappings")

        return self._self_edge_cache[ds_id]

    def _filter_self_edges(
        self, data: np.ndarray, ds: xr.Dataset, exclude_self_edges: bool = True
    ) -> tuple[np.ndarray, dict]:
        """Filter self-edges from mobility data.

        Args:
            data: 3D array of shape (time, home, destination)
            ds: xarray Dataset for self-edge mapping
            exclude_self_edges: If True, remove self-edges; if False, keep all data

        Returns:
            Tuple of (filtered_data, self_edge_stats) where:
            - filtered_data: Data with self-edges removed (flattened, non-zero only)
            - self_edge_stats: Dictionary with self-edge statistics
        """
        self_edge_pairs = self._get_self_edge_mapping(ds)

        if not exclude_self_edges:
            # Return all non-zero data
            nonzero_data = data[data > 0]
            return nonzero_data, {
                "self_edge_count": 0,
                "self_edge_total": 0,
                "self_edge_ratio": 0.0,
            }

        # Extract self-edge values
        self_edge_values = []
        for t in range(data.shape[0]):  # For each time step
            for home_idx, dest_idx in self_edge_pairs:
                val = data[t, home_idx, dest_idx]
                if val > 0:
                    self_edge_values.append(val)

        # Create mask for self-edges
        mask = np.ones_like(data, dtype=bool)
        for t in range(data.shape[0]):
            for home_idx, dest_idx in self_edge_pairs:
                mask[t, home_idx, dest_idx] = False

        # Apply mask and get non-zero mobility data
        filtered_data = data[mask]
        mobility_data = filtered_data[filtered_data > 0]

        # Calculate self-edge statistics
        self_edge_stats = {
            "self_edge_count": len(self_edge_values),
            "self_edge_total": sum(self_edge_values) if self_edge_values else 0,
            "self_edge_ratio": sum(self_edge_values) / data[data > 0].sum()
            if self_edge_values and data[data > 0].sum() > 0
            else 0.0,
            "self_edge_mean": np.mean(self_edge_values) if self_edge_values else 0.0,
        }

        return mobility_data, self_edge_stats

    def compute_basic_statistics(
        self,
        datasets: Dict[str, xr.Dataset],
        exclude_self_edges: bool = True,
        normalize_by_days: bool = True,
    ) -> pd.DataFrame:
        """Compute basic statistics for each month with improved methodology.

        Args:
            datasets: Dictionary of month -> xarray Dataset
            exclude_self_edges: If True, exclude self-edges (staying home) from analysis
            normalize_by_days: If True, normalize statistics by number of days in period

        Returns:
            DataFrame with statistics for each month
        """
        stats_list = []

        for month, ds in datasets.items():
            data = ds.person_hours.values  # Shape: (time, home, destination)
            num_days = data.shape[0]

            # Get time range for this dataset
            time_values = ds.time.values
            start_date = pd.Timestamp(time_values[0])
            end_date = pd.Timestamp(time_values[-1])

            # Filter self-edges if requested
            filtered_data, self_edge_stats = self._filter_self_edges(
                data, ds, exclude_self_edges
            )

            if len(filtered_data) == 0:
                logger.warning(f"No data remaining for {month} after filtering")
                continue

            # Calculate base statistics
            total_flow = filtered_data.sum()
            mean_weight = filtered_data.mean()
            median_weight = np.median(filtered_data)

            # Apply daily normalization if requested
            if normalize_by_days:
                daily_total_flow = total_flow / num_days
                daily_mean_weight = (
                    mean_weight  # Mean per edge doesn't change with normalization
                )
            else:
                daily_total_flow = total_flow
                daily_mean_weight = mean_weight

            stats_dict = {
                # Period information
                "month": month,
                "num_days": num_days,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "analysis_type": "mobility_only"
                if exclude_self_edges
                else "full_dataset",
                # Self-edge information
                "self_edge_ratio": self_edge_stats["self_edge_ratio"],
                "self_edge_count": self_edge_stats["self_edge_count"],
                "self_edge_mean": self_edge_stats.get("self_edge_mean", 0.0),
                # Raw statistics
                "total_values": data.size,
                "active_edges": len(filtered_data),
                "sparsity": 1 - len(filtered_data) / data.size,
                "total_flow": total_flow,
                "daily_total_flow": daily_total_flow,
                # Distribution statistics
                "min_weight": filtered_data.min(),
                "max_weight": filtered_data.max(),
                "mean_weight": mean_weight,
                "daily_mean_weight": daily_mean_weight,
                "median_weight": median_weight,
                "std_weight": filtered_data.std(),
                "skewness": stats.skew(filtered_data),
                "kurtosis": stats.kurtosis(filtered_data),
                # Percentiles
                "p10": np.percentile(filtered_data, 10),
                "p25": np.percentile(filtered_data, 25),
                "p75": np.percentile(filtered_data, 75),
                "p90": np.percentile(filtered_data, 90),
                "p95": np.percentile(filtered_data, 95),
                "p99": np.percentile(filtered_data, 99),
                # Log-scale metrics
                "log_range": np.log10(filtered_data.max())
                - np.log10(filtered_data.min()),
                "log_std": np.std(np.log10(filtered_data)),
                "cv": filtered_data.std()
                / filtered_data.mean(),  # coefficient of variation
            }
            stats_list.append(stats_dict)

            logger.info(
                f"Processed {month}: {num_days} days, "
                f"{'mobility-only' if exclude_self_edges else 'full dataset'} "
                f"({len(filtered_data):,} active edges, "
                f"{self_edge_stats['self_edge_ratio']:.1%} self-edges filtered)"
            )

        return pd.DataFrame(stats_list)

    def classify_lockdown_period(self, month_str: str) -> str:
        """Classify a month as pre/during/post lockdown or between lockdowns."""
        month_date = pd.Timestamp(f"{month_str}-15")  # Use mid-month date

        # Check if in lockdown periods
        for start, end in self.lockdown_periods:
            if start <= month_date <= end:
                return "during_lockdown"

        # Classify relative to lockdowns
        first_lockdown_start = self.lockdown_periods[0][0]
        first_lockdown_end = self.lockdown_periods[0][1]
        second_lockdown_start = self.lockdown_periods[1][0]

        if month_date < first_lockdown_start:
            return "pre_lockdown"
        elif first_lockdown_end < month_date < second_lockdown_start:
            return "between_lockdowns"
        else:
            return "post_lockdown"

    def classify_restriction_level(self, month_str: str) -> dict:
        """Classify a month by restriction severity level using new constants."""
        month_date = pd.Timestamp(f"{month_str}-15")  # Use mid-month date

        # Find matching restriction period
        for period_name, period_info in RESTRICTION_LEVELS.items():
            if period_info["start"] <= month_date <= period_info["end"]:
                return {
                    "period": period_name,
                    "severity": period_info["severity"],
                    "description": period_info["description"],
                }

        # Default if no period matches
        return {
            "period": "unknown",
            "severity": 0,
            "description": "Period not classified",
        }

    def classify_emergency_type(self, month_str: str) -> str:
        """Distinguish between full lockdown and curfew periods."""
        month_date = pd.Timestamp(f"{month_str}-15")

        # Check first state of emergency (full lockdown)
        first_start, first_end = STATE_OF_EMERGENCY["first"]["period"]
        if first_start <= month_date <= first_end:
            return "full_lockdown"

        # Check second state of emergency (curfew restrictions)
        second_start, second_end = STATE_OF_EMERGENCY["second"]["period"]
        if second_start <= month_date <= second_end:
            return "curfew_restrictions"

        return "no_emergency"

    def analyze_lockdown_impact(self, stats_df: pd.DataFrame) -> Dict:
        """Analyze the impact of lockdowns on mobility patterns."""
        # Add multiple classification methods
        stats_df["lockdown_period"] = stats_df["month"].apply(
            self.classify_lockdown_period
        )
        stats_df["emergency_type"] = stats_df["month"].apply(
            self.classify_emergency_type
        )

        # Add restriction level analysis
        restriction_data = stats_df["month"].apply(self.classify_restriction_level)
        stats_df["restriction_period"] = restriction_data.apply(lambda x: x["period"])
        stats_df["severity_level"] = restriction_data.apply(lambda x: x["severity"])
        stats_df["restriction_description"] = restriction_data.apply(
            lambda x: x["description"]
        )

        impact_analysis = {}

        # Group statistics by lockdown period using daily normalized metrics
        grouped = (
            stats_df.groupby("lockdown_period")
            .agg(
                {
                    "daily_mean_weight": [
                        "mean",
                        "std",
                    ],  # Use daily normalized metrics
                    "median_weight": ["mean", "std"],
                    "daily_total_flow": ["mean", "std"],  # Daily total flow
                    "sparsity": ["mean", "std"],
                    "cv": ["mean", "std"],
                    "log_range": ["mean", "std"],
                    "num_days": ["mean", "min", "max"],  # Track period lengths
                    "self_edge_ratio": ["mean", "std"],  # Self-edge filtering info
                }
            )
            .round(2)
        )

        # Group by emergency type (distinguishing full lockdown vs curfew)
        emergency_grouped = (
            stats_df.groupby("emergency_type")
            .agg(
                {
                    "daily_mean_weight": ["mean", "std"],
                    "median_weight": ["mean", "std"],
                    "daily_total_flow": ["mean", "std"],
                    "sparsity": ["mean", "std"],
                    "cv": ["mean", "std"],
                    "log_range": ["mean", "std"],
                    "num_days": ["mean", "min", "max"],
                    "self_edge_ratio": ["mean", "std"],
                }
            )
            .round(2)
        )

        # Group by severity level for gradient analysis
        severity_grouped = (
            stats_df.groupby("severity_level")
            .agg(
                {
                    "daily_mean_weight": ["mean", "std"],
                    "median_weight": ["mean", "std"],
                    "daily_total_flow": ["mean", "std"],
                    "sparsity": ["mean", "std"],
                    "cv": ["mean", "std"],
                    "log_range": ["mean", "std"],
                    "num_days": ["mean", "min", "max"],
                    "self_edge_ratio": ["mean", "std"],
                }
            )
            .round(2)
        )

        # Calculate percentage changes from baseline (pre-lockdown)
        if "pre_lockdown" in grouped.index:
            baseline_mean = grouped.loc["pre_lockdown", ("daily_mean_weight", "mean")]
            baseline_flow = grouped.loc["pre_lockdown", ("daily_total_flow", "mean")]
            baseline_sparsity = grouped.loc["pre_lockdown", ("sparsity", "mean")]

            for period in grouped.index:
                if period != "pre_lockdown":
                    period_mean = grouped.loc[period, ("daily_mean_weight", "mean")]
                    period_flow = grouped.loc[period, ("daily_total_flow", "mean")]
                    period_sparsity = grouped.loc[period, ("sparsity", "mean")]

                    impact_analysis[f"{period}_mean_change"] = (
                        (period_mean - baseline_mean) / baseline_mean * 100
                    )
                    impact_analysis[f"{period}_flow_change"] = (
                        (period_flow - baseline_flow) / baseline_flow * 100
                    )
                    impact_analysis[f"{period}_sparsity_change"] = (
                        (period_sparsity - baseline_sparsity) / baseline_sparsity * 100
                    )

        impact_analysis["period_stats"] = grouped
        impact_analysis["emergency_stats"] = emergency_grouped
        impact_analysis["severity_stats"] = severity_grouped
        impact_analysis["monthly_classification"] = stats_df[
            [
                "month",
                "lockdown_period",
                "emergency_type",
                "restriction_period",
                "severity_level",
                "restriction_description",
            ]
        ]

        return impact_analysis

    def analyze_lockdown_decay(self, datasets: Dict[str, xr.Dataset]) -> Dict:
        """Analyze the decaying effectiveness of lockdowns over time."""
        decay_analysis = {}

        # Process each lockdown period
        for lockdown_idx, (start_date, end_date) in enumerate(self.lockdown_periods, 1):
            lockdown_data = []

            # Collect daily data for this lockdown period
            for month_key, ds in datasets.items():
                month_date = pd.Timestamp(f"{month_key}-15")

                # Check if this month overlaps with current lockdown
                month_start = pd.Timestamp(f"{month_key}-01")
                month_end = (
                    pd.Timestamp(f"{month_key}-01")
                    + pd.DateOffset(months=1)
                    - pd.DateOffset(days=1)
                )

                if not (month_end < start_date or month_start > end_date):
                    # This month overlaps with lockdown
                    data = ds.person_hours.values
                    filtered_data, self_edge_stats = self._filter_self_edges(
                        data, ds, exclude_self_edges=True
                    )

                    if len(filtered_data) > 0:
                        lockdown_data.append(
                            {
                                "date": month_date,
                                "mean_weight": filtered_data.mean(),
                                "median_weight": np.median(filtered_data),
                                "sparsity": 1 - len(filtered_data) / data.size,
                                "total_flow": filtered_data.sum(),
                                "active_edges": len(filtered_data),
                            }
                        )

            if not lockdown_data:
                continue

            # Convert to DataFrame and sort by date
            lockdown_df = pd.DataFrame(lockdown_data).sort_values("date")

            # Calculate weeks since lockdown start
            lockdown_df["weeks_since_start"] = (
                (lockdown_df["date"] - start_date).dt.days / 7
            ).astype(int)

            # Get pre-lockdown baseline (average of 2 months before)
            baseline_data = []
            baseline_start = start_date - pd.DateOffset(months=2)

            for month_key, ds in datasets.items():
                month_date = pd.Timestamp(f"{month_key}-15")
                if baseline_start <= month_date < start_date:
                    data = ds.person_hours.values
                    filtered_data, _ = self._filter_self_edges(
                        data, ds, exclude_self_edges=True
                    )
                    if len(filtered_data) > 0:
                        baseline_data.append(filtered_data.mean())

            if baseline_data:
                baseline_mean = np.mean(baseline_data)
            else:
                # Use first available pre-lockdown month
                baseline_mean = None
                for month_key in sorted(datasets.keys()):
                    month_date = pd.Timestamp(f"{month_key}-15")
                    if month_date < start_date:
                        data = datasets[month_key].person_hours.values
                        filtered_data, _ = self._filter_self_edges(
                            data, datasets[month_key], exclude_self_edges=True
                        )
                        if len(filtered_data) > 0:
                            baseline_mean = filtered_data.mean()
                            break

            if baseline_mean is None:
                continue

            # Calculate effectiveness decay (% change from baseline)
            lockdown_df["effectiveness_index"] = (
                (lockdown_df["mean_weight"] - baseline_mean) / baseline_mean * 100
            )

            # Group by week and calculate weekly averages
            weekly_stats = (
                lockdown_df.groupby("weeks_since_start")
                .agg(
                    {
                        "mean_weight": "mean",
                        "effectiveness_index": "mean",
                        "sparsity": "mean",
                        "total_flow": "mean",
                    }
                )
                .reset_index()
            )

            # Calculate decay metrics
            if len(weekly_stats) > 1:
                # Fit linear regression to effectiveness over time
                from scipy import stats as scipy_stats

                weeks = weekly_stats["weeks_since_start"].values
                effectiveness = weekly_stats["effectiveness_index"].values

                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    weeks, effectiveness
                )

                # Find adaptation point (where effectiveness starts declining significantly)
                # Using rolling window to detect trend changes
                if len(effectiveness) > 3:
                    rolling_diff = pd.Series(effectiveness).diff().rolling(2).mean()
                    adaptation_week = None
                    for i, diff in enumerate(rolling_diff):
                        if (
                            i > 2 and diff > 0
                        ):  # Effectiveness starting to increase (less effective)
                            adaptation_week = i
                            break
                else:
                    adaptation_week = None

                # Calculate half-life (weeks to 50% effectiveness loss)
                initial_effect = weekly_stats.iloc[0]["effectiveness_index"]
                half_effect = initial_effect / 2
                half_life = None

                for _, row in weekly_stats.iterrows():
                    if (
                        initial_effect < 0 and row["effectiveness_index"] > half_effect
                    ) or (
                        initial_effect > 0 and row["effectiveness_index"] < half_effect
                    ):
                        half_life = row["weeks_since_start"]
                        break

                decay_analysis[f"lockdown_{lockdown_idx}"] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_days": (end_date - start_date).days,
                    "duration_weeks": len(weekly_stats),
                    "baseline_mean": baseline_mean,
                    "initial_effectiveness": weekly_stats.iloc[0][
                        "effectiveness_index"
                    ],
                    "final_effectiveness": weekly_stats.iloc[-1]["effectiveness_index"],
                    "decay_rate_per_week": slope,
                    "decay_r_squared": r_value**2,
                    "adaptation_week": adaptation_week,
                    "half_life_weeks": half_life,
                    "weekly_stats": weekly_stats,
                    "raw_data": lockdown_df,
                }

        return decay_analysis

    def analyze_intra_lockdown_patterns(self, decay_analysis: Dict) -> Dict:
        """Analyze patterns within each lockdown period (early, middle, late phases)."""
        intra_patterns = {}

        for lockdown_key, lockdown_data in decay_analysis.items():
            if "weekly_stats" not in lockdown_data:
                continue

            weekly_stats = lockdown_data["weekly_stats"]
            n_weeks = len(weekly_stats)

            if n_weeks < 3:
                continue

            # Divide lockdown into three phases
            phase_size = n_weeks // 3
            early_phase = weekly_stats.iloc[:phase_size]
            middle_phase = weekly_stats.iloc[phase_size : 2 * phase_size]
            late_phase = weekly_stats.iloc[2 * phase_size :]

            # Calculate phase statistics
            phases_analysis = {
                "early": {
                    "weeks": early_phase["weeks_since_start"].tolist(),
                    "mean_effectiveness": early_phase["effectiveness_index"].mean(),
                    "mean_weight": early_phase["mean_weight"].mean(),
                    "effectiveness_change": early_phase["effectiveness_index"].iloc[-1]
                    - early_phase["effectiveness_index"].iloc[0]
                    if len(early_phase) > 1
                    else 0,
                },
                "middle": {
                    "weeks": middle_phase["weeks_since_start"].tolist(),
                    "mean_effectiveness": middle_phase["effectiveness_index"].mean(),
                    "mean_weight": middle_phase["mean_weight"].mean(),
                    "effectiveness_change": middle_phase["effectiveness_index"].iloc[-1]
                    - middle_phase["effectiveness_index"].iloc[0]
                    if len(middle_phase) > 1
                    else 0,
                },
                "late": {
                    "weeks": late_phase["weeks_since_start"].tolist(),
                    "mean_effectiveness": late_phase["effectiveness_index"].mean(),
                    "mean_weight": late_phase["mean_weight"].mean(),
                    "effectiveness_change": late_phase["effectiveness_index"].iloc[-1]
                    - late_phase["effectiveness_index"].iloc[0]
                    if len(late_phase) > 1
                    else 0,
                },
            }

            # Calculate fatigue indicators
            fatigue_indicators = {
                "early_to_late_decay": phases_analysis["late"]["mean_effectiveness"]
                - phases_analysis["early"]["mean_effectiveness"],
                "acceleration_of_decay": phases_analysis["late"]["effectiveness_change"]
                - phases_analysis["early"]["effectiveness_change"],
                "middle_phase_stability": abs(
                    phases_analysis["middle"]["effectiveness_change"]
                ),
            }

            # Determine if lockdown showed fatigue pattern
            shows_fatigue = (
                fatigue_indicators["early_to_late_decay"] > 10  # Significant decay
                or fatigue_indicators["acceleration_of_decay"] > 5  # Accelerating decay
            )

            intra_patterns[lockdown_key] = {
                "phases": phases_analysis,
                "fatigue_indicators": fatigue_indicators,
                "shows_fatigue_pattern": shows_fatigue,
                "total_weeks": n_weeks,
            }

        return intra_patterns

    def create_visualizations(self, datasets: Dict[str, xr.Dataset], output_dir: Path):
        """Create comprehensive visualizations of edge weight distributions."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

        # 1. Combined histogram (linear and log scale)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Edge Weight Distribution Analysis", fontsize=16)

        all_nonzero_data = []
        monthly_data = {}

        # Collect all data (using filtered mobility-only data)
        for month, ds in datasets.items():
            data = ds.person_hours.values
            filtered_data, _ = self._filter_self_edges(
                data, ds, exclude_self_edges=True
            )
            all_nonzero_data.extend(filtered_data)
            monthly_data[month] = filtered_data

        all_nonzero_data = np.array(all_nonzero_data)

        # Linear scale histogram
        axes[0, 0].hist(all_nonzero_data, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("Edge Weight (person-hours)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution (Linear Scale)")
        axes[0, 0].set_xlim(0, np.percentile(all_nonzero_data, 99))

        # Log scale histogram
        log_data = np.log10(all_nonzero_data)
        axes[0, 1].hist(log_data, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("Log10(Edge Weight)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution (Log Scale)")

        # Box plot by month
        box_data = [
            monthly_data[month] for month in sorted(monthly_data.keys())[:6]
        ]  # First 6 months
        box_labels = sorted(monthly_data.keys())[:6]

        bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
        axes[1, 0].set_ylabel("Edge Weight (person-hours)")
        axes[1, 0].set_title("Monthly Distributions (Linear)")
        axes[1, 0].set_yscale("log")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # CDF
        sorted_data = np.sort(all_nonzero_data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].semilogx(sorted_data, y)
        axes[1, 1].set_xlabel("Edge Weight (person-hours)")
        axes[1, 1].set_ylabel("Cumulative Probability")
        axes[1, 1].set_title("Cumulative Distribution Function")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            output_dir / "edge_weight_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Temporal analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Temporal Analysis of Edge Weights", fontsize=16)

        months = sorted(monthly_data.keys())
        means = [np.mean(monthly_data[m]) for m in months]
        medians = [np.median(monthly_data[m]) for m in months]
        p95s = [np.percentile(monthly_data[m], 95) for m in months]
        # Calculate sparsities considering only mobility (inter-regional) flows
        sparsities = []
        for m in months:
            data = datasets[m].person_hours.values
            total_possible_flows = data.size
            active_mobility_flows = len(
                monthly_data[m]
            )  # Already filtered for mobility
            sparsity = 1 - active_mobility_flows / total_possible_flows
            sparsities.append(sparsity)

        # Mean over time
        axes[0, 0].plot(months, means, "o-", label="Mean")
        axes[0, 0].plot(months, medians, "s-", label="Median")
        axes[0, 0].set_ylabel("Edge Weight")
        axes[0, 0].set_title("Central Tendencies Over Time")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 95th percentile over time
        axes[0, 1].plot(months, p95s, "o-", color="red")
        axes[0, 1].set_ylabel("95th Percentile Weight")
        axes[0, 1].set_title("Heavy Tail Behavior Over Time")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Sparsity over time
        axes[1, 0].plot(months, sparsities, "o-", color="green")
        axes[1, 0].set_ylabel("Sparsity Ratio")
        axes[1, 0].set_title("Network Sparsity Over Time")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Log-scale standard deviation
        log_stds = [np.std(np.log10(monthly_data[m])) for m in months]
        axes[1, 1].plot(months, log_stds, "o-", color="purple")
        axes[1, 1].set_ylabel("Log10 Standard Deviation")
        axes[1, 1].set_title("Distribution Spread Over Time")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "temporal_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Learning separability analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Model Learning Separability Analysis", fontsize=16)

        # Dynamic range analysis
        dynamic_ranges = []
        for month in months:
            data = monthly_data[month]
            dr = np.log10(data.max() / data.min())
            dynamic_ranges.append(dr)

        axes[0].bar(range(len(months)), dynamic_ranges)
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Log10(Max/Min) Dynamic Range")
        axes[0].set_title("Dynamic Range per Month")
        axes[0].set_xticks(range(len(months)))
        axes[0].set_xticklabels(months, rotation=45)

        # Coefficient of variation
        cvs = [monthly_data[m].std() / monthly_data[m].mean() for m in months]
        axes[1].bar(range(len(months)), cvs)
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Coefficient of Variation")
        axes[1].set_title("Relative Variability per Month")
        axes[1].set_xticks(range(len(months)))
        axes[1].set_xticklabels(months, rotation=45)

        # Quantile separation
        sep_ratios = []
        for month in months:
            data = monthly_data[month]
            p75 = np.percentile(data, 75)
            p25 = np.percentile(data, 25)
            separation = p75 / p25  # Interquartile ratio
            sep_ratios.append(separation)

        axes[2].bar(range(len(months)), sep_ratios)
        axes[2].set_xlabel("Month")
        axes[2].set_ylabel("P75/P25 Ratio")
        axes[2].set_title("Quartile Separation")
        axes[2].set_xticks(range(len(months)))
        axes[2].set_xticklabels(months, rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_dir / "learning_separability.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    def create_lockdown_visualizations(
        self, stats_df: pd.DataFrame, impact_analysis: Dict, output_dir: Path
    ):
        """Create visualizations specifically focused on lockdown impact analysis."""
        # 4. Lockdown impact analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("COVID-19 Lockdown Impact on Mobility Networks", fontsize=16)

        # Add lockdown period classification
        stats_df_with_lockdown = stats_df.copy()
        stats_df_with_lockdown["lockdown_period"] = stats_df_with_lockdown[
            "month"
        ].apply(self.classify_lockdown_period)

        # Color mapping for lockdown periods
        period_colors = {
            "pre_lockdown": "#2E86AB",
            "during_lockdown": "#F24236",
            "between_lockdowns": "#F6AE2D",
            "post_lockdown": "#2F9B69",
        }

        # Timeline of mean weights with lockdown periods
        months = stats_df_with_lockdown["month"].tolist()
        mean_weights = stats_df_with_lockdown["daily_mean_weight"].tolist()
        colors = [
            period_colors[period]
            for period in stats_df_with_lockdown["lockdown_period"]
        ]

        axes[0, 0].bar(range(len(months)), mean_weights, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel("Month")
        axes[0, 0].set_ylabel("Mean Edge Weight (person-hours)")
        axes[0, 0].set_title("Timeline: Mean Edge Weights")
        axes[0, 0].set_xticks(range(len(months)))
        axes[0, 0].set_xticklabels(months, rotation=45)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=period.replace("_", " ").title())
            for period, color in period_colors.items()
        ]
        axes[0, 0].legend(handles=legend_elements, loc="upper right")

        # Sparsity changes over time
        sparsity_values = stats_df_with_lockdown["sparsity"].tolist()
        axes[0, 1].bar(range(len(months)), sparsity_values, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Network Sparsity")
        axes[0, 1].set_title("Timeline: Network Sparsity")
        axes[0, 1].set_xticks(range(len(months)))
        axes[0, 1].set_xticklabels(months, rotation=45)

        # Coefficient of variation
        cv_values = stats_df_with_lockdown["cv"].tolist()
        axes[0, 2].bar(range(len(months)), cv_values, color=colors, alpha=0.7)
        axes[0, 2].set_xlabel("Month")
        axes[0, 2].set_ylabel("Coefficient of Variation")
        axes[0, 2].set_title("Timeline: Weight Variability")
        axes[0, 2].set_xticks(range(len(months)))
        axes[0, 2].set_xticklabels(months, rotation=45)

        # Box plot comparison by lockdown period
        period_order = [
            "pre_lockdown",
            "during_lockdown",
            "between_lockdowns",
            "post_lockdown",
        ]
        available_periods = [
            p
            for p in period_order
            if p in stats_df_with_lockdown["lockdown_period"].values
        ]

        mean_weight_by_period = [
            stats_df_with_lockdown[stats_df_with_lockdown["lockdown_period"] == period][
                "daily_mean_weight"
            ].tolist()
            for period in available_periods
        ]

        bp = axes[1, 0].boxplot(
            mean_weight_by_period,
            patch_artist=True,
            labels=[p.replace("_", " ").title() for p in available_periods],
        )
        for patch, period in zip(bp["boxes"], available_periods):
            patch.set_facecolor(period_colors[period])
            patch.set_alpha(0.7)
        axes[1, 0].set_ylabel("Mean Edge Weight")
        axes[1, 0].set_title("Mean Weight Distribution by Period")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Percentage changes from baseline
        if "pre_lockdown" in stats_df_with_lockdown["lockdown_period"].values:
            baseline_stats = stats_df_with_lockdown[
                stats_df_with_lockdown["lockdown_period"] == "pre_lockdown"
            ]
            baseline_mean = baseline_stats["daily_mean_weight"].mean()
            baseline_sparsity = baseline_stats["sparsity"].mean()

            period_changes = {}
            for period in available_periods:
                if period != "pre_lockdown":
                    period_stats = stats_df_with_lockdown[
                        stats_df_with_lockdown["lockdown_period"] == period
                    ]
                    period_mean = period_stats["daily_mean_weight"].mean()
                    period_sparsity = period_stats["sparsity"].mean()

                    period_changes[period] = {
                        "mean_change": (
                            (period_mean - baseline_mean) / baseline_mean * 100
                        ),
                        "sparsity_change": (
                            (period_sparsity - baseline_sparsity)
                            / baseline_sparsity
                            * 100
                        ),
                    }

            # Plot percentage changes
            periods = list(period_changes.keys())
            mean_changes = [period_changes[p]["mean_change"] for p in periods]
            sparsity_changes = [period_changes[p]["sparsity_change"] for p in periods]

            x_pos = np.arange(len(periods))
            width = 0.35

            bars1 = axes[1, 1].bar(
                x_pos - width / 2,
                mean_changes,
                width,
                label="Mean Weight Change (%)",
                alpha=0.7,
            )
            bars2 = axes[1, 1].bar(
                x_pos + width / 2,
                sparsity_changes,
                width,
                label="Sparsity Change (%)",
                alpha=0.7,
            )

            axes[1, 1].set_xlabel("Lockdown Period")
            axes[1, 1].set_ylabel("Percentage Change from Baseline")
            axes[1, 1].set_title("Impact vs Pre-Lockdown Baseline")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(
                [p.replace("_", " ").title() for p in periods], rotation=45
            )
            axes[1, 1].legend()
            axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.3)

        # Recovery analysis - show trend from peak disruption
        peak_disruption_month = stats_df_with_lockdown.loc[
            stats_df_with_lockdown["daily_mean_weight"].idxmax(), "month"
        ]
        peak_idx = months.index(peak_disruption_month)

        recovery_weights = mean_weights[peak_idx:]
        recovery_months = months[peak_idx:]

        axes[1, 2].plot(
            range(len(recovery_weights)),
            recovery_weights,
            "o-",
            linewidth=2,
            markersize=6,
        )
        axes[1, 2].set_xlabel("Months since Peak Disruption")
        axes[1, 2].set_ylabel("Mean Edge Weight")
        axes[1, 2].set_title(f"Recovery Pattern from {peak_disruption_month}")
        axes[1, 2].set_xticks(range(len(recovery_months)))
        axes[1, 2].set_xticklabels(recovery_months, rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_dir / "lockdown_impact_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Lockdown visualizations saved to {output_dir}")

    def create_decay_visualizations(
        self, decay_analysis: Dict, intra_patterns: Dict, output_dir: Path
    ):
        """Create visualizations for lockdown effectiveness decay analysis."""
        if not decay_analysis:
            logger.warning("No decay analysis data available for visualization")
            return

        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            "Lockdown Effectiveness Decay Analysis", fontsize=18, fontweight="bold"
        )

        # Color scheme
        lockdown_colors = ["#E63946", "#457B9D"]  # Red for first, blue for second
        phase_colors = {"early": "#2E7D32", "middle": "#FFA726", "late": "#C62828"}

        # 1. Weekly effectiveness over time for both lockdowns
        ax1 = fig.add_subplot(gs[0, :2])

        for idx, (lockdown_key, lockdown_data) in enumerate(decay_analysis.items()):
            if "weekly_stats" not in lockdown_data:
                continue

            weekly_stats = lockdown_data["weekly_stats"]
            weeks = weekly_stats["weeks_since_start"]
            effectiveness = weekly_stats["effectiveness_index"]

            lockdown_num = lockdown_key.split("_")[1]
            color = lockdown_colors[idx % 2]

            # Plot actual data
            ax1.plot(
                weeks,
                effectiveness,
                "o-",
                color=color,
                linewidth=2,
                markersize=6,
                label=f"Lockdown {lockdown_num}",
                alpha=0.8,
            )

            # Add trend line
            if lockdown_data["decay_rate_per_week"] is not None:
                z = np.polyfit(weeks, effectiveness, 1)
                p = np.poly1d(z)
                ax1.plot(weeks, p(weeks), "--", color=color, alpha=0.5, linewidth=1.5)

            # Mark adaptation point
            if lockdown_data["adaptation_week"]:
                adapt_week = lockdown_data["adaptation_week"]
                adapt_effect = weekly_stats.iloc[adapt_week]["effectiveness_index"]
                ax1.scatter(
                    adapt_week,
                    adapt_effect,
                    s=150,
                    color=color,
                    marker="v",
                    edgecolor="black",
                    linewidth=2,
                    zorder=5,
                )
                ax1.annotate(
                    f"Adaptation\nWeek {adapt_week}",
                    xy=(adapt_week, adapt_effect),
                    xytext=(adapt_week + 1, adapt_effect - 5),
                    fontsize=9,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3, label="Baseline")
        ax1.set_xlabel("Weeks Since Lockdown Start", fontsize=12)
        ax1.set_ylabel("Effectiveness Index (% Change from Baseline)", fontsize=12)
        ax1.set_title(
            "Weekly Lockdown Effectiveness Trajectory", fontsize=14, fontweight="bold"
        )
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # 2. Decay rate comparison
        ax2 = fig.add_subplot(gs[0, 2])

        decay_rates = []
        lockdown_labels = []
        colors = []

        for idx, (lockdown_key, lockdown_data) in enumerate(decay_analysis.items()):
            if (
                "decay_rate_per_week" in lockdown_data
                and lockdown_data["decay_rate_per_week"] is not None
            ):
                decay_rates.append(lockdown_data["decay_rate_per_week"])
                lockdown_num = lockdown_key.split("_")[1]
                lockdown_labels.append(f"Lockdown {lockdown_num}")
                colors.append(lockdown_colors[idx % 2])

        if decay_rates:
            bars = ax2.bar(
                range(len(decay_rates)), decay_rates, color=colors, alpha=0.7
            )
            ax2.set_ylabel("Decay Rate (% per week)", fontsize=12)
            ax2.set_title("Effectiveness Decay Rate", fontsize=14, fontweight="bold")
            ax2.set_xticks(range(len(lockdown_labels)))
            ax2.set_xticklabels(lockdown_labels)

            # Add value labels on bars
            for bar, rate in zip(bars, decay_rates):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.2f}%/week",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                )

        # 3. Phase-wise effectiveness (early, middle, late)
        ax3 = fig.add_subplot(gs[1, :2])

        if intra_patterns:
            x_positions = []
            x_labels = []
            bar_width = 0.25

            for lockdown_idx, (lockdown_key, pattern_data) in enumerate(
                intra_patterns.items()
            ):
                if "phases" not in pattern_data:
                    continue

                phases = pattern_data["phases"]
                lockdown_num = lockdown_key.split("_")[1]

                base_x = lockdown_idx * 4

                for phase_idx, phase_name in enumerate(["early", "middle", "late"]):
                    x_pos = base_x + phase_idx * bar_width
                    effectiveness = phases[phase_name]["mean_effectiveness"]

                    ax3.bar(
                        x_pos,
                        effectiveness,
                        bar_width,
                        color=phase_colors[phase_name],
                        alpha=0.7,
                        label=phase_name.capitalize() if lockdown_idx == 0 else "",
                    )

                    # Add value label
                    ax3.text(
                        x_pos,
                        effectiveness,
                        f"{effectiveness:.1f}%",
                        ha="center",
                        va="bottom" if effectiveness > 0 else "top",
                        fontsize=9,
                    )

                # Add lockdown label
                x_labels.append(f"Lockdown {lockdown_num}")
                x_positions.append(base_x + bar_width)

            ax3.set_xlabel("Lockdown Period", fontsize=12)
            ax3.set_ylabel("Mean Effectiveness Index (%)", fontsize=12)
            ax3.set_title(
                "Phase-wise Effectiveness Analysis", fontsize=14, fontweight="bold"
            )
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels(x_labels)
            ax3.axhline(y=0, color="black", linestyle="--", alpha=0.3)
            ax3.legend(loc="best")
            ax3.grid(True, alpha=0.3, axis="y")

        # 4. Fatigue indicators
        ax4 = fig.add_subplot(gs[1, 2])

        if intra_patterns:
            fatigue_data = []
            labels = []

            for lockdown_key, pattern_data in intra_patterns.items():
                if "fatigue_indicators" in pattern_data:
                    lockdown_num = lockdown_key.split("_")[1]
                    fatigue = pattern_data["fatigue_indicators"]
                    fatigue_data.append(
                        [
                            fatigue["early_to_late_decay"],
                            fatigue["acceleration_of_decay"],
                            fatigue["middle_phase_stability"],
                        ]
                    )
                    labels.append(f"L{lockdown_num}")

            if fatigue_data:
                fatigue_data = np.array(fatigue_data).T
                indicator_names = [
                    "Early→Late\nDecay",
                    "Acceleration\nof Decay",
                    "Middle Phase\nInstability",
                ]

                x = np.arange(len(indicator_names))
                width = 0.35

                for i, label in enumerate(labels):
                    offset = width * (i - len(labels) / 2 + 0.5)
                    bars = ax4.bar(
                        x + offset,
                        fatigue_data[:, i],
                        width,
                        label=label,
                        color=lockdown_colors[i],
                        alpha=0.7,
                    )

                    # Add value labels
                    for bar, value in zip(bars, fatigue_data[:, i]):
                        ax4.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            value,
                            f"{value:.1f}",
                            ha="center",
                            va="bottom" if value > 0 else "top",
                            fontsize=9,
                        )

                ax4.set_ylabel("Fatigue Indicator Value", fontsize=12)
                ax4.set_title(
                    "Lockdown Fatigue Indicators", fontsize=14, fontweight="bold"
                )
                ax4.set_xticks(x)
                ax4.set_xticklabels(indicator_names)
                ax4.legend()
                ax4.axhline(y=0, color="black", linestyle="--", alpha=0.3)
                ax4.grid(True, alpha=0.3, axis="y")

        # 5. Comparison metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("tight")
        ax5.axis("off")

        # Create comparison table
        table_data = []
        headers = ["Metric", "Lockdown 1", "Lockdown 2", "Difference"]

        if len(decay_analysis) >= 2:
            l1_data = list(decay_analysis.values())[0]
            l2_data = list(decay_analysis.values())[1]

            metrics = [
                ("Duration (days)", l1_data["duration_days"], l2_data["duration_days"]),
                (
                    "Duration (weeks)",
                    l1_data["duration_weeks"],
                    l2_data["duration_weeks"],
                ),
                (
                    "Initial Effect (%)",
                    l1_data["initial_effectiveness"],
                    l2_data["initial_effectiveness"],
                ),
                (
                    "Final Effect (%)",
                    l1_data["final_effectiveness"],
                    l2_data["final_effectiveness"],
                ),
                (
                    "Total Decay (%)",
                    l1_data["final_effectiveness"] - l1_data["initial_effectiveness"],
                    l2_data["final_effectiveness"] - l2_data["initial_effectiveness"],
                ),
                (
                    "Decay Rate (%/week)",
                    l1_data["decay_rate_per_week"],
                    l2_data["decay_rate_per_week"],
                ),
                ("R² of Decay", l1_data["decay_r_squared"], l2_data["decay_r_squared"]),
                (
                    "Adaptation Week",
                    l1_data["adaptation_week"],
                    l2_data["adaptation_week"],
                ),
                (
                    "Half-life (weeks)",
                    l1_data["half_life_weeks"],
                    l2_data["half_life_weeks"],
                ),
            ]

            for metric_name, val1, val2 in metrics:
                if val1 is not None and val2 is not None:
                    if isinstance(val1, (int, float)):
                        diff = val2 - val1
                        if (
                            "days" in metric_name
                            or "weeks" in metric_name
                            or "Week" in metric_name
                        ):
                            table_data.append(
                                [
                                    metric_name,
                                    f"{val1:.0f}",
                                    f"{val2:.0f}",
                                    f"{diff:+.0f}",
                                ]
                            )
                        else:
                            table_data.append(
                                [
                                    metric_name,
                                    f"{val1:.2f}",
                                    f"{val2:.2f}",
                                    f"{diff:+.2f}",
                                ]
                            )
                else:
                    table_data.append(
                        [
                            metric_name,
                            f"{val1:.2f}" if val1 is not None else "N/A",
                            f"{val2:.2f}" if val2 is not None else "N/A",
                            "N/A",
                        ]
                    )

        if table_data:
            table = ax5.table(
                cellText=table_data,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                colWidths=[0.3, 0.2, 0.2, 0.2],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#40466e")
                table[(0, i)].set_text_props(weight="bold", color="white")

            # Color code differences
            for i in range(1, len(table_data) + 1):
                if table_data[i - 1][3] != "N/A":
                    try:
                        diff_val = float(table_data[i - 1][3])
                        if (
                            "Decay" in table_data[i - 1][0]
                            or "Half-life" in table_data[i - 1][0]
                        ):
                            # For decay metrics, higher is worse
                            if diff_val > 0:
                                table[(i, 3)].set_facecolor("#ffcccc")  # Light red
                            elif diff_val < 0:
                                table[(i, 3)].set_facecolor("#ccffcc")  # Light green
                        else:
                            # For other metrics, context-dependent
                            if abs(diff_val) > 0:
                                table[(i, 3)].set_facecolor("#ffffcc")  # Light yellow
                    except ValueError:
                        pass

            ax5.set_title(
                "Lockdown Comparison Metrics", fontsize=14, fontweight="bold", pad=20
            )

        plt.suptitle(
            "Lockdown Effectiveness Decay Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "lockdown_decay_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Decay visualizations saved to {output_dir}")

    def generate_report(
        self,
        stats_df: pd.DataFrame,
        impact_analysis: Dict,
        output_file: Path,
        decay_analysis: Dict = None,
        intra_patterns: Dict = None,
    ):
        """Generate a comprehensive markdown report including lockdown impact and decay analysis."""
        report = []
        report.append("# Edge Weight Distribution Analysis Report\n")
        report.append(
            "Analysis of OD network edge weight distributions for model learning assessment.\n"
        )

        # Add methodology section
        report.append("## Methodology\n")
        analysis_type = (
            "mobility-only"
            if stats_df["analysis_type"].iloc[0] == "mobility_only"
            else "full dataset"
        )
        if stats_df["analysis_type"].iloc[0] == "mobility_only":
            avg_self_edge_ratio = stats_df["self_edge_ratio"].mean()
            report.append(f"**Analysis Type**: {analysis_type} (self-edges filtered)")
            report.append(
                f"- **Self-edge filtering**: Removed {avg_self_edge_ratio:.1%} of flows representing 'staying home' patterns"
            )
            report.append("- **Focus**: Inter-regional mobility patterns only")
        else:
            report.append(f"**Analysis Type**: {analysis_type} (includes all flows)")
            report.append(
                "- **Coverage**: All origin-destination flows including self-edges"
            )

        report.append(
            "\n**Period Normalization**: Daily averages used for fair temporal comparisons"
        )
        period_lengths = stats_df.groupby("month")["num_days"].first()
        min_days, max_days = period_lengths.min(), period_lengths.max()
        report.append(
            f"- **Period variation**: {min_days} to {max_days} days per dataset"
        )
        report.append("- **Normalization**: All statistics converted to daily averages")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary\n")
        avg_sparsity = stats_df["sparsity"].mean()
        avg_log_range = stats_df["log_range"].mean()
        avg_cv = stats_df["cv"].mean()

        report.append(
            f"- **Average sparsity**: {avg_sparsity:.1%} (high sparsity network)"
        )
        report.append(
            f"- **Average dynamic range**: {avg_log_range:.1f} orders of magnitude"
        )
        report.append(f"- **Average coefficient of variation**: {avg_cv:.1f}")
        report.append(f"- **Number of months analyzed**: {len(stats_df)}")
        report.append("")

        # Key Findings
        report.append("## Key Findings for Model Learning\n")

        if avg_log_range > 4:
            report.append(
                "✅ **Excellent separation**: >4 orders of magnitude range provides strong signal differentiation"
            )
        elif avg_log_range > 2:
            report.append(
                "⚠️ **Moderate separation**: 2-4 orders of magnitude may require careful normalization"
            )
        else:
            report.append(
                "❌ **Poor separation**: <2 orders of magnitude may challenge neighborhood weight learning"
            )

        if avg_sparsity > 0.9:
            report.append(
                "⚠️ **High sparsity**: >90% zero values may require sparse graph techniques"
            )
        elif avg_sparsity > 0.7:
            report.append(
                "✅ **Moderate sparsity**: 70-90% sparsity is manageable for GNN training"
            )
        else:
            report.append(
                "✅ **Low sparsity**: <70% sparsity is ideal for dense graph learning"
            )

        if avg_cv > 5:
            report.append(
                "✅ **High variability**: Strong variation supports neighborhood weight differentiation"
            )
        else:
            report.append(
                "⚠️ **Low variability**: Limited variation may reduce learning signal strength"
            )

        report.append("")

        # Statistical Summary
        report.append("## Statistical Summary by Month\n")
        report.append(
            "| Month | Days | Sparsity | Log Range | Mean Weight | Daily Flow | CV | Self-Edge % |"
        )
        report.append(
            "|-------|------|----------|-----------|-------------|------------|----|-----------| "
        )

        for _, row in stats_df.iterrows():
            report.append(
                f"| {row['month']} | {row['num_days']} | {row['sparsity']:.1%} | {row['log_range']:.1f} | "
                f"{row['daily_mean_weight']:.0f} | {row['daily_total_flow']:,.0f} | "
                f"{row['cv']:.1f} | {row['self_edge_ratio']:.1%} |"
            )

        report.append("")

        # Distribution Characteristics
        report.append("## Distribution Characteristics\n")
        report.append("### Percentile Analysis")
        report.append("Average percentile values across all months:\n")

        percentiles = ["p10", "p25", "p75", "p90", "p95", "p99"]
        for p in percentiles:
            avg_val = stats_df[p].mean()
            report.append(f"- **{p.upper()}**: {avg_val:.0f} person-hours")

        report.append("")

        # Lockdown Impact Analysis
        report.append("## COVID-19 Lockdown Impact Analysis\n")
        report.append(
            "**Updated Analysis**: Using corrected Spanish lockdown timeline based on official government sources."
        )
        report.append(
            "First lockdown: March 15 - June 21, 2020 (was June 20). Second period: October 25, 2020 - May 9, 2021 (curfew restrictions, not full lockdown).\n"
        )

        if "period_stats" in impact_analysis:
            period_stats = impact_analysis["period_stats"]

            # Summary of lockdown effects
            report.append("### Key Lockdown Findings\n")

            # Find periods with highest changes
            baseline_mean = None
            if "pre_lockdown" in period_stats.index:
                baseline_mean = period_stats.loc[
                    "pre_lockdown", ("daily_mean_weight", "mean")
                ]

                report.append(
                    f"**Baseline (Pre-lockdown)**: {baseline_mean:.0f} person-hours mean weight (daily normalized, mobility-only)"
                )

                for period in period_stats.index:
                    if period != "pre_lockdown":
                        period_mean = period_stats.loc[
                            period, ("daily_mean_weight", "mean")
                        ]
                        change_pct = (period_mean - baseline_mean) / baseline_mean * 100
                        period_name = period.replace("_", " ").title()

                        if change_pct > 10:
                            report.append(
                                f"- **{period_name}**: {period_mean:.0f} person-hours (+{change_pct:.1f}% increase)"
                            )
                        elif change_pct < -10:
                            report.append(
                                f"- **{period_name}**: {period_mean:.0f} person-hours ({change_pct:.1f}% decrease)"
                            )
                        else:
                            report.append(
                                f"- **{period_name}**: {period_mean:.0f} person-hours ({change_pct:+.1f}% change)"
                            )

            report.append("")
            report.append(
                "### Counter-Intuitive Pattern: Increased Edge Weights During Lockdowns"
            )
            report.append(
                "Unlike typical expectations, mean edge weights **increased** during lockdown periods."
            )
            report.append("This suggests a **network concentration effect** where:")
            report.append(
                "- Fewer origin-destination pairs remained active (higher sparsity)"
            )
            report.append(
                "- Remaining active connections carried higher traffic intensity"
            )
            report.append(
                "- Essential trips became more concentrated on specific routes"
            )

            report.append("")
            report.append("### Network Structural Changes")

            if "pre_lockdown" in period_stats.index:
                baseline_sparsity = period_stats.loc[
                    "pre_lockdown", ("sparsity", "mean")
                ]

                for period in ["during_lockdown", "between_lockdowns", "post_lockdown"]:
                    if period in period_stats.index:
                        period_sparsity = period_stats.loc[period, ("sparsity", "mean")]
                        sparsity_change = (
                            (period_sparsity - baseline_sparsity)
                            / baseline_sparsity
                            * 100
                        )
                        period_name = period.replace("_", " ").title()

                        if sparsity_change > 1:
                            report.append(
                                f"- **{period_name}**: {period_sparsity:.1%} sparsity (+{sparsity_change:.1f}% from baseline)"
                            )
                        else:
                            report.append(
                                f"- **{period_name}**: {period_sparsity:.1%} sparsity ({sparsity_change:+.1f}% from baseline)"
                            )

            report.append("")
            report.append("### Statistical Summary by Lockdown Period")
            report.append(
                "| Period | Mean Weight | Daily Flow | Sparsity | CV | Log Range | Days |"
            )
            report.append(
                "|--------|-------------|------------|----------|----|-----------|----- |"
            )

            for period in period_stats.index:
                period_name = period.replace("_", " ").title()
                mean_w = period_stats.loc[period, ("daily_mean_weight", "mean")]
                daily_flow = period_stats.loc[period, ("daily_total_flow", "mean")]
                sparsity = period_stats.loc[period, ("sparsity", "mean")]
                cv = period_stats.loc[period, ("cv", "mean")]
                log_range = period_stats.loc[period, ("log_range", "mean")]
                num_days = period_stats.loc[period, ("num_days", "mean")]

                report.append(
                    f"| {period_name} | {mean_w:.0f} | {daily_flow:,.0f} | {sparsity:.1%} | {cv:.1f} | {log_range:.1f} | {num_days:.0f} |"
                )

            report.append("")

        # Recommendations
        report.append("## Recommendations for Model Training\n")

        report.append("### Data Preprocessing")
        if avg_log_range > 4:
            report.append(
                "- Consider log-normalization to handle extreme dynamic range"
            )
        report.append(
            "- Apply minimum threshold filtering to reduce noise from very small flows"
        )
        if avg_sparsity > 0.9:
            report.append(
                "- Use sparse graph representations to handle high sparsity efficiently"
            )

        report.append("\n### Model Architecture")
        if avg_cv > 5:
            report.append(
                "- Attention mechanisms should work well with high variability"
            )
        report.append(
            "- Consider adaptive aggregation to handle heterogeneous edge weights"
        )
        report.append(
            "- Normalize edge weights within local neighborhoods for stable training"
        )

        report.append("\n### Training Considerations")
        report.append("- Use gradient clipping due to heavy-tailed distributions")
        report.append(
            "- Consider weighted loss functions to handle imbalanced edge importance"
        )
        report.append("- Monitor for overfitting on high-weight edges")

        report.append("")

        # Lockdown Decay Analysis
        if decay_analysis:
            report.append("## Lockdown Effectiveness Decay Analysis\n")
            report.append(
                "Advanced analysis of how lockdown effectiveness changed over time, revealing"
            )
            report.append("adaptation patterns and fatigue effects.\n")

            # Summary of decay findings
            report.append("### Key Decay Findings\n")

            for lockdown_key, lockdown_data in decay_analysis.items():
                lockdown_num = lockdown_key.split("_")[1]
                duration_days = lockdown_data["duration_days"]
                duration_weeks = lockdown_data["duration_weeks"]

                report.append(
                    f"**Lockdown {lockdown_num}** ({lockdown_data['start_date'].strftime('%Y-%m-%d')} to {lockdown_data['end_date'].strftime('%Y-%m-%d')}):"
                )
                report.append(
                    f"- Duration: {duration_days} days ({duration_weeks} weeks)"
                )
                report.append(
                    f"- Initial effectiveness: {lockdown_data['initial_effectiveness']:.1f}% change from baseline"
                )
                report.append(
                    f"- Final effectiveness: {lockdown_data['final_effectiveness']:.1f}% change from baseline"
                )

                total_change = (
                    lockdown_data["final_effectiveness"]
                    - lockdown_data["initial_effectiveness"]
                )
                report.append(
                    f"- Total effectiveness change: {total_change:+.1f}% over lockdown period"
                )

                if lockdown_data["decay_rate_per_week"] is not None:
                    report.append(
                        f"- Decay rate: {lockdown_data['decay_rate_per_week']:.2f}% per week"
                    )
                    report.append(
                        f"- Decay model fit (R²): {lockdown_data['decay_r_squared']:.3f}"
                    )

                if lockdown_data["adaptation_week"] is not None:
                    report.append(
                        f"- Adaptation detected at week {lockdown_data['adaptation_week']}"
                    )
                    adaptation_days = lockdown_data["adaptation_week"] * 7
                    report.append(
                        f"  (approximately {adaptation_days} days into lockdown)"
                    )
                else:
                    report.append("- No clear adaptation point detected")

                if lockdown_data["half_life_weeks"] is not None:
                    report.append(
                        f"- Half-life of effectiveness: {lockdown_data['half_life_weeks']:.1f} weeks"
                    )
                else:
                    report.append("- Half-life not reached during lockdown period")

                report.append("")

            # Comparative analysis
            if len(decay_analysis) >= 2:
                report.append("### Comparative Lockdown Analysis\n")

                lockdowns = list(decay_analysis.values())
                l1, l2 = lockdowns[0], lockdowns[1]

                # Duration comparison
                duration_diff = l2["duration_days"] - l1["duration_days"]
                report.append(
                    f"- **Duration difference**: Lockdown 2 was {abs(duration_diff)} days {'longer' if duration_diff > 0 else 'shorter'}"
                )

                # Effectiveness comparison
                if (
                    l1["initial_effectiveness"] is not None
                    and l2["initial_effectiveness"] is not None
                ):
                    init_diff = (
                        l2["initial_effectiveness"] - l1["initial_effectiveness"]
                    )
                    report.append(
                        f"- **Initial effectiveness**: Lockdown 2 started {abs(init_diff):.1f}% {'more' if init_diff > 0 else 'less'} effective"
                    )

                # Decay rate comparison
                if (
                    l1["decay_rate_per_week"] is not None
                    and l2["decay_rate_per_week"] is not None
                ):
                    decay_diff = l2["decay_rate_per_week"] - l1["decay_rate_per_week"]
                    if abs(decay_diff) > 0.1:
                        report.append(
                            f"- **Decay rate**: Lockdown 2 decayed {abs(decay_diff):.2f}% per week {'faster' if decay_diff > 0 else 'slower'}"
                        )

                        if decay_diff > 0:
                            report.append(
                                "  This suggests **lockdown fatigue** - people adapted faster to the second lockdown"
                            )
                        else:
                            report.append(
                                "  This suggests the second lockdown maintained effectiveness better"
                            )
                    else:
                        report.append(
                            "- **Decay rate**: Similar decay rates between both lockdowns"
                        )

                # Adaptation timing
                if (
                    l1["adaptation_week"] is not None
                    and l2["adaptation_week"] is not None
                ):
                    adapt_diff = l2["adaptation_week"] - l1["adaptation_week"]
                    report.append(
                        f"- **Adaptation timing**: People adapted {abs(adapt_diff)} weeks {'earlier' if adapt_diff < 0 else 'later'} in Lockdown 2"
                    )

                    if adapt_diff < 0:
                        report.append(
                            "  This indicates **learning effect** - faster adaptation to repeated restrictions"
                        )

                report.append("")

            # Phase analysis
            if intra_patterns:
                report.append("### Intra-Lockdown Phase Analysis\n")
                report.append(
                    "Each lockdown was divided into early, middle, and late phases to analyze"
                )
                report.append(
                    "how effectiveness changed throughout the restriction period.\n"
                )

                for lockdown_key, pattern_data in intra_patterns.items():
                    lockdown_num = lockdown_key.split("_")[1]
                    phases = pattern_data["phases"]
                    fatigue = pattern_data["fatigue_indicators"]

                    report.append(f"**Lockdown {lockdown_num} Phase Analysis:**")
                    report.append(
                        f"- Early phase ({len(phases['early']['weeks'])} weeks): {phases['early']['mean_effectiveness']:.1f}% average effectiveness"
                    )
                    report.append(
                        f"- Middle phase ({len(phases['middle']['weeks'])} weeks): {phases['middle']['mean_effectiveness']:.1f}% average effectiveness"
                    )
                    report.append(
                        f"- Late phase ({len(phases['late']['weeks'])} weeks): {phases['late']['mean_effectiveness']:.1f}% average effectiveness"
                    )

                    # Fatigue assessment
                    early_to_late = fatigue["early_to_late_decay"]
                    if early_to_late > 15:
                        report.append(
                            f"- **Strong fatigue pattern**: {early_to_late:.1f}% effectiveness decline from early to late phase"
                        )
                    elif early_to_late > 5:
                        report.append(
                            f"- **Moderate fatigue pattern**: {early_to_late:.1f}% effectiveness decline from early to late phase"
                        )
                    else:
                        report.append(
                            f"- **Stable effectiveness**: Only {early_to_late:.1f}% change from early to late phase"
                        )

                    if pattern_data["shows_fatigue_pattern"]:
                        report.append(
                            "- ⚠️ **Fatigue detected**: Clear signs of decreasing effectiveness over time"
                        )
                    else:
                        report.append(
                            "- ✅ **Sustained effectiveness**: No significant fatigue pattern detected"
                        )

                    report.append("")

                # Cross-lockdown fatigue comparison
                if len(intra_patterns) >= 2:
                    l1_fatigue = list(intra_patterns.values())[0][
                        "shows_fatigue_pattern"
                    ]
                    l2_fatigue = list(intra_patterns.values())[1][
                        "shows_fatigue_pattern"
                    ]

                    if l1_fatigue and l2_fatigue:
                        report.append(
                            "**Overall Pattern**: Both lockdowns showed fatigue effects, suggesting"
                        )
                        report.append(
                            "that long-duration restrictions lose effectiveness over time regardless"
                        )
                        report.append("of when they occur.\n")
                    elif not l1_fatigue and l2_fatigue:
                        report.append(
                            "**Overall Pattern**: Only the second lockdown showed fatigue, suggesting"
                        )
                        report.append(
                            "**lockdown adaptation** - people became more resistant to restrictions.\n"
                        )
                    elif l1_fatigue and not l2_fatigue:
                        report.append(
                            "**Overall Pattern**: Only the first lockdown showed fatigue, suggesting"
                        )
                        report.append(
                            "lessons learned improved the effectiveness of the second lockdown.\n"
                        )
                    else:
                        report.append(
                            "**Overall Pattern**: Neither lockdown showed significant fatigue,"
                        )
                        report.append(
                            "suggesting both maintained effectiveness throughout their duration.\n"
                        )

            report.append("### Policy Implications\n")

            # Generate policy recommendations based on findings
            recommendations = []

            if decay_analysis:
                avg_adaptation_weeks = []
                for lockdown_data in decay_analysis.values():
                    if lockdown_data["adaptation_week"] is not None:
                        avg_adaptation_weeks.append(lockdown_data["adaptation_week"])

                if avg_adaptation_weeks:
                    avg_adapt = np.mean(avg_adaptation_weeks)
                    recommendations.append(
                        "- **Optimal lockdown duration**: Based on adaptation patterns, effectiveness"
                    )
                    recommendations.append(
                        f"  begins declining around week {avg_adapt:.1f}. Consider {int(avg_adapt * 7)}-day lockdowns"
                    )
                    recommendations.append(
                        "  with reassessment points rather than extended periods."
                    )

            if intra_patterns:
                fatigue_count = sum(
                    1 for p in intra_patterns.values() if p["shows_fatigue_pattern"]
                )
                if fatigue_count > 0:
                    recommendations.append(
                        "- **Fatigue mitigation**: Implement graduated restrictions or"
                    )
                    recommendations.append(
                        "  intermittent enforcement to maintain effectiveness"
                    )
                    recommendations.append(
                        "- **Public communication**: Increase messaging frequency during"
                    )
                    recommendations.append(
                        "  middle and late phases when fatigue typically emerges"
                    )

            if len(decay_analysis) >= 2:
                l1_decay = list(decay_analysis.values())[0]["decay_rate_per_week"]
                l2_decay = list(decay_analysis.values())[1]["decay_rate_per_week"]

                if (
                    l1_decay is not None
                    and l2_decay is not None
                    and l2_decay > l1_decay
                ):
                    recommendations.append(
                        "- **Learning effect**: Population adapted faster to second lockdown."
                    )
                    recommendations.append(
                        "  Future restrictions may have diminishing returns and require"
                    )
                    recommendations.append(
                        "  alternative approaches or stronger enforcement"
                    )

            for rec in recommendations:
                report.append(rec)

            report.append("")

        # Visualizations
        report.append("## Visualizations\n")
        report.append("The following plots are generated with this analysis:\n")
        report.append(
            "1. `edge_weight_distributions.png` - Histograms, box plots, and CDF"
        )
        report.append(
            "2. `temporal_analysis.png` - Time series of distribution properties"
        )
        report.append(
            "3. `learning_separability.png` - Metrics relevant to model learning"
        )
        report.append(
            "4. `lockdown_impact_analysis.png` - COVID-19 lockdown impact visualizations"
        )
        if decay_analysis:
            report.append(
                "5. `lockdown_decay_analysis.png` - Lockdown effectiveness decay analysis"
            )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(report))

        logger.info(f"Report saved to {output_file}")

    def run_analysis(
        self,
        output_dir: str = "reports/edge_weight_analysis",
        exclude_self_edges: bool = True,
        normalize_by_days: bool = True,
    ):
        """Run the complete edge weight analysis with improved methodology.

        Args:
            output_dir: Output directory for results
            exclude_self_edges: If True, analyze only inter-regional mobility (recommended)
            normalize_by_days: If True, normalize statistics by period length (recommended)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        analysis_type = "mobility_only" if exclude_self_edges else "full_dataset"
        normalization = "daily_normalized" if normalize_by_days else "raw_totals"
        logger.info(
            f"Starting edge weight distribution analysis: {analysis_type}, {normalization}"
        )

        # Load data
        datasets = self.load_all_data()
        if not datasets:
            raise ValueError("No datasets loaded")

        # Compute statistics with new methodology
        stats_df = self.compute_basic_statistics(
            datasets, exclude_self_edges, normalize_by_days
        )
        stats_df.to_csv(
            output_path / f"edge_weight_statistics_{analysis_type}_{normalization}.csv",
            index=False,
        )

        # Analyze lockdown impact
        impact_analysis = self.analyze_lockdown_impact(stats_df)

        # Analyze lockdown decay
        logger.info("Analyzing lockdown effectiveness decay...")
        decay_analysis = self.analyze_lockdown_decay(datasets)

        # Analyze intra-lockdown patterns
        logger.info("Analyzing intra-lockdown patterns...")
        intra_patterns = self.analyze_intra_lockdown_patterns(decay_analysis)

        # Create visualizations
        self.create_visualizations(datasets, output_path)
        self.create_lockdown_visualizations(stats_df, impact_analysis, output_path)
        self.create_decay_visualizations(decay_analysis, intra_patterns, output_path)

        # Generate report
        self.generate_report(
            stats_df,
            impact_analysis,
            output_path / "edge_weight_analysis_report.md",
            decay_analysis,
            intra_patterns,
        )

        logger.info(f"Analysis complete. Results saved to {output_path}")
        return stats_df, impact_analysis, decay_analysis, intra_patterns


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    analyzer = EdgeWeightAnalyzer()
    stats_df, impact_analysis, decay_analysis, intra_patterns = analyzer.run_analysis()

    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Analyzed {len(stats_df)} months of OD data")
    print(f"Average sparsity: {stats_df['sparsity'].mean():.1%}")
    print(
        f"Average dynamic range: {stats_df['log_range'].mean():.1f} orders of magnitude"
    )
    print(f"Average coefficient of variation: {stats_df['cv'].mean():.1f}")

    # Quick assessment for model learning
    avg_log_range = stats_df["log_range"].mean()
    avg_cv = stats_df["cv"].mean()

    print("\n=== MODEL LEARNING ASSESSMENT ===")
    if avg_log_range > 4 and avg_cv > 3:
        print(
            "✅ EXCELLENT: Strong separation and variability support neighborhood weight learning"
        )
    elif avg_log_range > 2 and avg_cv > 1:
        print(
            "✅ GOOD: Sufficient separation for model to learn meaningful edge weights"
        )
    else:
        print("⚠️ CHALLENGING: Limited separation may require careful preprocessing")

    # Lockdown impact summary
    print("\n=== COVID-19 LOCKDOWN IMPACT ===")
    if "period_stats" in impact_analysis:
        period_stats = impact_analysis["period_stats"]

        if "pre_lockdown" in period_stats.index:
            baseline_mean = period_stats.loc[
                "pre_lockdown", ("daily_mean_weight", "mean")
            ]
            print(
                f"Pre-lockdown baseline: {baseline_mean:.0f} person-hours (daily, mobility-only)"
            )

            if "during_lockdown" in period_stats.index:
                lockdown_mean = period_stats.loc[
                    "during_lockdown", ("daily_mean_weight", "mean")
                ]
                change_pct = (lockdown_mean - baseline_mean) / baseline_mean * 100
                print(
                    f"During lockdown: {lockdown_mean:.0f} person-hours ({change_pct:+.1f}% change)"
                )

                if change_pct > 10:
                    print(
                        "⚠️ COUNTER-INTUITIVE: Edge weights INCREASED during lockdowns"
                    )
                    print(
                        "   This suggests network concentration effects on remaining active routes"
                    )
                elif change_pct < -10:
                    print("✅ EXPECTED: Edge weights decreased during lockdowns")
                else:
                    print("📊 MINIMAL IMPACT: Edge weights remained relatively stable")

    # Decay analysis summary
    print("\n=== LOCKDOWN DECAY ANALYSIS ===")
    if decay_analysis:
        for lockdown_key, lockdown_data in decay_analysis.items():
            lockdown_num = lockdown_key.split("_")[1]
            print(f"\nLockdown {lockdown_num}:")
            print(
                f"Duration: {lockdown_data['duration_days']} days ({lockdown_data['duration_weeks']} weeks)"
            )

            if lockdown_data["decay_rate_per_week"] is not None:
                print(
                    f"Decay rate: {lockdown_data['decay_rate_per_week']:.2f}% per week"
                )

                if abs(lockdown_data["decay_rate_per_week"]) > 1:
                    print(
                        "⚠️ SIGNIFICANT DECAY: Lockdown effectiveness changed substantially over time"
                    )
                elif abs(lockdown_data["decay_rate_per_week"]) > 0.5:
                    print("📈 MODERATE DECAY: Some decline in effectiveness detected")
                else:
                    print(
                        "✅ STABLE: Lockdown maintained effectiveness throughout period"
                    )

            if lockdown_data["adaptation_week"] is not None:
                adaptation_days = lockdown_data["adaptation_week"] * 7
                print(
                    f"Adaptation point: Week {lockdown_data['adaptation_week']} (~{adaptation_days} days)"
                )

                if lockdown_data["adaptation_week"] <= 4:
                    print("⚡ EARLY ADAPTATION: People adapted quickly to restrictions")
                elif lockdown_data["adaptation_week"] <= 8:
                    print("📊 MODERATE ADAPTATION: Normal adaptation timeline")
                else:
                    print("🛡️ SLOW ADAPTATION: Restrictions remained effective longer")

        # Compare lockdowns
        if len(decay_analysis) >= 2:
            l1_decay = list(decay_analysis.values())[0]["decay_rate_per_week"]
            l2_decay = list(decay_analysis.values())[1]["decay_rate_per_week"]

            if l1_decay is not None and l2_decay is not None:
                print("\n🔍 LOCKDOWN COMPARISON:")
                decay_diff = l2_decay - l1_decay

                if abs(decay_diff) > 0.5:
                    if decay_diff > 0:
                        print(
                            f"Lockdown 2 decayed {decay_diff:.2f}% per week FASTER → Lockdown fatigue detected"
                        )
                    else:
                        print(
                            f"Lockdown 2 decayed {abs(decay_diff):.2f}% per week SLOWER → Better sustained effectiveness"
                        )
                else:
                    print("Both lockdowns had similar decay patterns")
    else:
        print("No decay analysis data available")


if __name__ == "__main__":
    main()
