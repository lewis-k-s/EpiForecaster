"""Utilities to build region graph tensors from GeoJSON + population tables."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from libpysal import weights

logger = logging.getLogger(__name__)

DEFAULT_MOBILITY_WINDOW_DAYS = 30


@dataclass
class RegionGraphPreprocessConfig:
    geojson_path: Path = Path("data/files/fl_municipios_catalonia.geojson")
    population_csv_path: Path = Path("data/files/fl_population_por_municipis.csv")
    geojson_id_field: str = "id"
    population_id_field: str = "id"
    population_value_field: str = "d.population"
    output_path: Path = Path("outputs/region_graph/region_graph.zarr")
    contiguity: str = "queen"  # "queen" or "rook"
    metric_crs: str = "EPSG:3035"
    mobility_zarr_path: Path | None = Path("data/files/mobility.zarr")
    start_date: str | None = "2021-04-10"
    end_date: str | None = "2021-05-09"


class RegionGraphPreprocessor:
    def __init__(self, config: RegionGraphPreprocessConfig) -> None:
        self.config = config
        self._resolved_start_date: str | None = None
        self._resolved_end_date: str | None = None
        self._flow_source: str = "unknown"

    def run(self) -> dict[str, Any]:
        gdf = self._load_regions()
        df = self._join_population(gdf)
        features = self._build_features(df)
        region_ids = df[self.config.geojson_id_field].tolist()
        adjacency = self._build_contiguity(df)
        edge_index = self._edge_index_from_weights(adjacency, region_ids)
        flows = self._build_flow_matrix(edge_index, region_ids)

        output_path = self._write_zarr(features, edge_index, flows, region_ids)

        return {
            "output_path": str(output_path),
            "num_regions": len(region_ids),
            "feature_dim": features.shape[1],
            "num_edges": edge_index.shape[1],
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_regions(self) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(self.config.geojson_path)
        if self.config.geojson_id_field not in gdf.columns:
            raise ValueError(
                f"GeoJSON missing id column '{self.config.geojson_id_field}'. Available columns: {list(gdf.columns)}"
            )
        gdf = gdf[[self.config.geojson_id_field, "geometry"]].copy()
        gdf[self.config.geojson_id_field] = gdf[self.config.geojson_id_field].astype(
            str
        )
        if gdf.geometry.is_empty.any():
            raise ValueError(
                "GeoJSON contains empty geometries; clean the file before preprocessing."
            )
        return gdf

    def _join_population(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        population = pd.read_csv(self.config.population_csv_path)
        if self.config.population_id_field not in population.columns:
            raise ValueError(
                f"Population CSV missing id column '{self.config.population_id_field}'. Columns: {list(population.columns)}"
            )
        if self.config.population_value_field not in population.columns:
            raise ValueError(
                f"Population CSV missing value column '{self.config.population_value_field}'. Columns: {list(population.columns)}"
            )

        pop_id_col = f"{self.config.population_id_field}_pop"
        population = population.rename(
            columns={self.config.population_id_field: pop_id_col}
        )
        population[pop_id_col] = population[pop_id_col].astype(str)

        merged = gdf.merge(
            population[[pop_id_col, self.config.population_value_field]],
            left_on=self.config.geojson_id_field,
            right_on=pop_id_col,
            how="left",
        )
        value_col = self.config.population_value_field
        if merged[value_col].isna().any():
            median_pop = merged[value_col].median(skipna=True)
            merged[value_col] = merged[value_col].fillna(
                median_pop if not np.isnan(median_pop) else 0.0
            )
        merged = merged.drop(columns=[pop_id_col])
        return merged

    # ------------------------------------------------------------------
    def _build_features(self, df: gpd.GeoDataFrame) -> np.ndarray:
        metric = df.to_crs(self.config.metric_crs)
        area = metric.geometry.area / 1_000_000.0
        perimeter = metric.geometry.length / 1_000.0
        centroid_metric = metric.geometry.centroid
        centroid_geo = gpd.GeoSeries(
            centroid_metric, crs=self.config.metric_crs
        ).to_crs("EPSG:4326")
        lon = centroid_geo.x
        lat = centroid_geo.y
        population = df[self.config.population_value_field].astype(float)
        density = np.divide(
            population,
            area.replace(0, np.nan),
            out=np.zeros_like(population, dtype=float),
            where=area.values != 0,
        )

        feature_matrix = np.vstack(
            [
                area.to_numpy(),
                perimeter.to_numpy(),
                population.to_numpy(),
                density,
                lon.to_numpy(),
                lat.to_numpy(),
            ]
        ).T.astype(np.float32)
        return feature_matrix

    def _build_contiguity(self, df: gpd.GeoDataFrame) -> weights.W:
        ids = df[self.config.geojson_id_field].tolist()
        if self.config.contiguity == "rook":
            w = weights.Rook.from_dataframe(df, ids=ids)
        else:
            w = weights.Queen.from_dataframe(df, ids=ids)
        return w

    def _edge_index_from_weights(
        self, w: weights.W, region_ids: list[str]
    ) -> np.ndarray:
        id_to_idx = {rid: idx for idx, rid in enumerate(region_ids)}
        edges: list[tuple[int, int]] = []
        for src_id, neighbors in w.neighbors.items():
            src_idx = id_to_idx[src_id]
            for dst_id in neighbors:
                dst_idx = id_to_idx[dst_id]
                edges.append((src_idx, dst_idx))

        if not edges:
            raise ValueError("No adjacency edges detected; check the GeoJSON validity.")

        edge_array = np.array(edges, dtype=np.int64)
        edge_array = np.unique(edge_array, axis=0)
        return edge_array.T

    def _build_flow_matrix(
        self, edge_index: np.ndarray, region_ids: list[str]
    ) -> np.ndarray:
        flows = self._compute_average_mobility_flows(region_ids)
        if flows is not None:
            self._flow_source = "mobility"
            return flows
        self._resolved_start_date = None
        self._resolved_end_date = None
        self._flow_source = "adjacency"
        return self._adjacency_flow_matrix(edge_index, len(region_ids))

    def _compute_average_mobility_flows(
        self, region_ids: list[str]
    ) -> np.ndarray | None:
        mobility_path = self.config.mobility_zarr_path
        if mobility_path is None:
            return None

        mobility_path = Path(mobility_path)
        if not mobility_path.exists():
            logger.warning(
                "Mobility dataset not found at %s; falling back to adjacency-implied flows",
                mobility_path,
            )
            return None

        try:
            with xr.open_zarr(str(mobility_path), decode_coords="all") as ds:
                var_name = self._detect_mobility_variable(ds)
                mobility = ds[var_name]
                rename_dims: dict[str, str] = {}
                if "date" in mobility.dims:
                    rename_dims["date"] = "time"
                if "source" in mobility.dims:
                    rename_dims["source"] = "origin"
                if "destination" in mobility.dims:
                    rename_dims["destination"] = "target"
                mobility = mobility.rename(rename_dims)

                for required_dim in ("time", "origin", "target"):
                    if required_dim not in mobility.dims:
                        raise ValueError(
                            f"Mobility dataset is missing required dimension '{required_dim}'"
                        )

                time_index = pd.DatetimeIndex(pd.to_datetime(mobility["time"].values))
                start_dt, end_dt = self._resolve_temporal_bounds(time_index)
                self._resolved_start_date = start_dt.date().isoformat()
                self._resolved_end_date = end_dt.date().isoformat()
                subset = mobility.sel(time=slice(start_dt, end_dt))
                if subset.sizes.get("time", 0) == 0:
                    raise ValueError(
                        f"No mobility records between {start_dt.date()} and {end_dt.date()}"
                    )

                subset = subset.assign_coords(
                    origin=self._normalize_region_ids(subset["origin"].values.tolist()),
                    target=self._normalize_region_ids(subset["target"].values.tolist()),
                )

                ordered_region_ids = self._normalize_region_ids(region_ids)
                available_origins = {
                    str(val).strip() for val in subset["origin"].values
                }
                missing = [
                    rid for rid in ordered_region_ids if rid not in available_origins
                ]
                if missing:
                    raise ValueError(
                        "Mobility dataset is missing regions required for graph preprocessing: "
                        + ", ".join(missing[:10])
                        + ("..." if len(missing) > 10 else "")
                    )

                subset = subset.sel(
                    origin=ordered_region_ids, target=ordered_region_ids
                )
                mean_flows = subset.mean(dim="time", skipna=True)
                flow_matrix = mean_flows.to_numpy().astype(np.float32)
                np.nan_to_num(flow_matrix, copy=False)
                return flow_matrix
        except Exception as exc:  # pragma: no cover - runtime data dependency
            logger.warning(
                "Failed to compute OD flows from %s (%s); using adjacency-implied flows",
                mobility_path,
                exc,
            )
            return None

    def _detect_mobility_variable(self, ds: xr.Dataset) -> str:
        for candidate in ("mobility", "trips", "origin_destination_matrix"):
            if candidate in ds:
                return candidate
        raise ValueError(
            "Mobility dataset must contain a 'mobility' or 'trips' variable"
        )

    def _resolve_temporal_bounds(
        self, dates: pd.DatetimeIndex
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if dates.empty:
            raise ValueError("Mobility dataset does not contain any timestamps")

        available_start = dates.min()
        available_end = dates.max()

        if self.config.end_date:
            end = pd.Timestamp(self.config.end_date)
        else:
            end = available_end
        if end > available_end:
            logger.warning(
                "Requested end date %s truncated to dataset max %s",
                end.date(),
                available_end.date(),
            )
            end = available_end
        if end < available_start:
            raise ValueError(
                f"Requested end date {end.date()} precedes available data ({available_start.date()})"
            )

        if self.config.start_date:
            start = pd.Timestamp(self.config.start_date)
        else:
            start = end - pd.Timedelta(days=DEFAULT_MOBILITY_WINDOW_DAYS - 1)
        if start < available_start:
            logger.warning(
                "Requested start date %s raised to dataset min %s",
                start.date(),
                available_start.date(),
            )
            start = available_start
        if start > end:
            raise ValueError(
                f"Requested start date {start.date()} exceeds end date {end.date()}"
            )

        return start.normalize(), end.normalize()

    @staticmethod
    def _normalize_region_ids(region_ids: Sequence[Any]) -> list[str]:
        formatted: list[str] = []
        for rid in region_ids:
            rid_str = str(rid).strip()
            if rid_str.isdigit():
                rid_str = rid_str.zfill(5)
            formatted.append(rid_str)
        return formatted

    @staticmethod
    def _adjacency_flow_matrix(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        flows = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_index.size == 0:
            return flows
        src, dst = edge_index
        flows[src, dst] = 1.0
        return flows

    # ------------------------------------------------------------------
    def _write_zarr(
        self,
        features: np.ndarray,
        edge_index: np.ndarray,
        flows: np.ndarray,
        region_ids: list[str],
    ) -> Path:
        output_path = self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            import shutil

            shutil.rmtree(output_path)
        root = zarr.open_group(str(output_path), mode="w", zarr_version=2)
        features_ds = root.create_dataset(
            "features", data=features, shape=features.shape, dtype="float32"
        )
        self._assign_dimensions(features_ds, ("region", "feature"))

        edge_index_ds = root.create_dataset(
            "edge_index", data=edge_index, shape=edge_index.shape, dtype="int64"
        )
        self._assign_dimensions(edge_index_ds, ("axis", "edge"))

        flows_ds = root.create_dataset(
            "flows", data=flows, shape=flows.shape, dtype="float32"
        )
        self._assign_dimensions(flows_ds, ("source", "target"))

        max_len = max(len(rid) for rid in region_ids)
        rid_array = np.array(region_ids, dtype=f"<U{max_len}")
        region_ids_ds = root.create_dataset(
            "region_ids", data=rid_array, shape=rid_array.shape, dtype=rid_array.dtype
        )
        self._assign_dimensions(region_ids_ds, ("region",))

        metadata = {
            "num_regions": len(region_ids),
            "feature_dim": features.shape[1],
            "num_edges": int(edge_index.shape[1]),
            "flow_source": self._flow_source,
            "geojson_path": str(self.config.geojson_path),
            "population_csv_path": str(self.config.population_csv_path),
            "geojson_id_field": self.config.geojson_id_field,
            "population_value_field": self.config.population_value_field,
            "mobility_zarr_path": str(self.config.mobility_zarr_path)
            if self.config.mobility_zarr_path
            else None,
            "mobility_start_date": self._resolved_start_date,
            "mobility_end_date": self._resolved_end_date,
        }
        root.attrs["metadata"] = json.dumps(metadata)
        return output_path

    @staticmethod
    def _assign_dimensions(zarr_array, dims: tuple[str, ...]) -> None:
        zarr_array.attrs["dimension_names"] = list(dims)
        zarr_array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
