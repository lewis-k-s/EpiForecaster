"""Region embedding trainer and configuration utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from graph.node_encoder import InductiveNodeEncoder
from models.region_losses import (
    CommunityOrientedLoss,
    SpatialContiguityPrior,
    SpatialOnlyLoss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    """Expand relative paths declared inside YAML configs relative to ``base_dir``."""

    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass
class RegionDataConfig:
    """Fields under the ``data`` YAML block for region training.

    Attributes:
        zarr_path: Path to zarr file containing features/edges/flows/ids.
        normalize_features: Whether to L2-normalize feature rows.
    """

    zarr_path: Path
    normalize_features: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path) -> "RegionDataConfig":
        """Instantiate from ``data`` dictionary while resolving relative paths."""

        zarr_path_str = raw.get("zarr_path")
        if not zarr_path_str:
            raise ValueError("Region training config requires zarr_path under `data`.")
        zarr_path = _resolve_path(base_dir, zarr_path_str)
        return cls(
            zarr_path=zarr_path,
            normalize_features=raw.get("normalize_features", True),
        )


@dataclass
class EncoderConfig:
    """Hyper-parameters under the ``encoder`` YAML block."""

    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    aggregation: str = "mean"
    residual: bool = False
    normalize: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EncoderConfig":
        raw = raw or {}
        defaults = cls()
        values = {
            f.name: raw.get(f.name, getattr(defaults, f.name)) for f in fields(cls)
        }
        return cls(**values)


@dataclass
class TrainingConfig:
    """Parameters under ``training`` controlling optimization and logging."""

    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    device: str = "auto"
    log_every: int = 10
    gradient_clip: float = 1.0
    seed: int | None = None
    checkpoint_every: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrainingConfig":
        raw = raw or {}
        defaults = cls()
        values = {
            f.name: raw.get(f.name, getattr(defaults, f.name)) for f in fields(cls)
        }
        return cls(**values)


@dataclass
class SamplingConfig:
    """Options under ``sampling`` describing pair counts and thresholds."""

    positive_pairs: int = 4096
    negative_pairs: int = 4096
    hop_pairs: int = 2048
    min_flow_threshold: float = 1.0
    hop_threshold: int = 2
    max_hops: int = 5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SamplingConfig":
        raw = raw or {}
        defaults = cls()
        values = {
            f.name: raw.get(f.name, getattr(defaults, f.name)) for f in fields(cls)
        }
        return cls(**values)


@dataclass
class LossConfig:
    """Specifies the ``loss`` block with its weighting hyper-parameters."""

    loss_type: str = "community"  # "community" or "spatial"
    spatial_weight: float = 1.0
    autocorr_weight: float = 0.5
    ratio_weight: float = 1.0
    hop_weight: float = 0.3
    temperature: float = 0.1
    margin: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LossConfig":
        raw = raw or {}
        defaults = cls()
        values = {
            f.name: raw.get(f.name, getattr(defaults, f.name)) for f in fields(cls)
        }
        return cls(**values)


@dataclass
class OutputConfig:
    """Artifacts listed under the ``output`` block."""

    output_dir: Path = Path("outputs/region_embeddings")
    embedding_filename: str = "region_embeddings.pt"
    metrics_filename: str = "region_training_metrics.json"
    cluster_labels_filename: str = "region_clusters.json"
    save_numpy: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, base_dir: Path) -> "OutputConfig":
        raw = raw or {}
        output_dir = _resolve_path(base_dir, raw.get("output_dir", cls.output_dir))
        return cls(
            output_dir=output_dir or cls.output_dir,
            embedding_filename=raw.get("embedding_filename", cls.embedding_filename),
            metrics_filename=raw.get("metrics_filename", cls.metrics_filename),
            cluster_labels_filename=raw.get(
                "cluster_labels_filename", cls.cluster_labels_filename
            ),
            save_numpy=raw.get("save_numpy", cls.save_numpy),
        )


@dataclass
class ClusteringConfig:
    """Controls the optional ``clustering`` post-processing step."""

    enabled: bool = True
    num_clusters: int = 14
    linkage: str = "ward"
    compute_connectivity: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ClusteringConfig":
        raw = raw or {}
        defaults = cls()
        values = {
            f.name: raw.get(f.name, getattr(defaults, f.name)) for f in fields(cls)
        }
        return cls(**values)


@dataclass
class RegionTrainerConfig:
    """Top-level structure representing the entire region-trainer YAML schema."""

    data: RegionDataConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    config_path: Path | None = None

    @classmethod
    def from_file(cls, config_path: str | Path) -> "RegionTrainerConfig":
        path = Path(config_path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw or {}, base_dir=path.parent, config_path=path)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], base_dir: Path, config_path: Path | None = None
    ) -> "RegionTrainerConfig":
        data_cfg = RegionDataConfig.from_dict(raw.get("data", {}), base_dir)
        encoder_cfg = EncoderConfig.from_dict(raw.get("encoder"))
        training_cfg = TrainingConfig.from_dict(raw.get("training"))
        sampling_cfg = SamplingConfig.from_dict(raw.get("sampling"))
        loss_cfg = LossConfig.from_dict(raw.get("loss"))
        output_cfg = OutputConfig.from_dict(raw.get("output"), base_dir)
        clustering_cfg = ClusteringConfig.from_dict(raw.get("clustering"))
        return cls(
            data=data_cfg,
            encoder=encoder_cfg,
            training=training_cfg,
            sampling=sampling_cfg,
            loss=loss_cfg,
            output=output_cfg,
            clustering=clustering_cfg,
            config_path=config_path,
        )

    def apply_cli_overrides(
        self,
        *,
        epochs: int | None = None,
        device: str | None = None,
        output_dir: Path | None = None,
        disable_clustering: bool = False,
    ) -> None:
        """Apply ad-hoc CLI overrides without mutating the backing YAML."""

        if epochs is not None:
            self.training.epochs = epochs
        if device is not None:
            self.training.device = device
        if output_dir is not None:
            self.output.output_dir = output_dir
        if disable_clustering:
            self.clustering.enabled = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize config back to primitive types for logging or dumps."""

        data_dict = asdict(self.data)
        data_dict["zarr_path"] = str(data_dict["zarr_path"])
        return {
            "data": data_dict,
            "encoder": asdict(self.encoder),
            "training": asdict(self.training),
            "sampling": asdict(self.sampling),
            "loss": asdict(self.loss),
            "output": {
                **asdict(self.output),
                "output_dir": str(self.output.output_dir),
            },
            "clustering": asdict(self.clustering),
        }


# ---------------------------------------------------------------------------
# Pair sampler utilities
# ---------------------------------------------------------------------------


class PairSampler:
    def __init__(
        self,
        flow_matrix: torch.Tensor | None,
        edge_index: torch.Tensor,
        hop_distances: torch.Tensor,
        sampling_config: SamplingConfig,
        rng: np.random.Generator,
    ) -> None:
        self.flow_matrix = (
            flow_matrix.detach().cpu().numpy() if flow_matrix is not None else None
        )
        self.edge_index = edge_index.detach().cpu().numpy()
        self.hop_distances = hop_distances.detach().cpu().numpy()
        self.sampling = sampling_config
        self.rng = rng
        self.num_nodes = hop_distances.size(0)
        self._prepare_pairs()

    def _prepare_pairs(self) -> None:
        if self.flow_matrix is not None:
            positive_mask = self.flow_matrix > self.sampling.min_flow_threshold
            negative_mask = self.flow_matrix <= self.sampling.min_flow_threshold
            if not np.any(positive_mask):
                max_flow = (
                    float(self.flow_matrix.max()) if self.flow_matrix.size else 0.0
                )
                raise ValueError(
                    "min_flow_threshold=%.2f filtered out all flows (max flow=%.2f). "
                    "Positive ratio samples will be empty; consider lowering the threshold.",
                    self.sampling.min_flow_threshold,
                    max_flow,
                )
            self.positive_pairs = np.argwhere(positive_mask)
            self.positive_flow = self.flow_matrix[positive_mask]
            self.negative_pairs = np.argwhere(negative_mask)
        else:
            # Use adjacency edges as positives when flows are unavailable
            self.positive_pairs = self.edge_index.T
            self.positive_flow = np.ones(self.positive_pairs.shape[0], dtype=np.float32)
            # Use hop-based negatives only
            self.negative_pairs = np.empty((0, 2), dtype=np.int64)

        hop_mask = self.hop_distances > self.sampling.hop_threshold
        hop_mask &= np.isfinite(self.hop_distances)
        if self.sampling.max_hops is not None:
            hop_mask &= self.hop_distances <= self.sampling.max_hops
        self.hop_pairs = np.argwhere(hop_mask)
        self.hop_values = self.hop_distances[hop_mask]

    def sample(self) -> dict[str, torch.Tensor]:
        pos_idx = self._sample_indices(
            self.positive_pairs.shape[0], self.sampling.positive_pairs
        )
        neg_idx = self._sample_indices(
            self.negative_pairs.shape[0], self.sampling.negative_pairs
        )
        hop_idx = self._sample_indices(self.hop_pairs.shape[0], self.sampling.hop_pairs)

        def gather(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
            if indices.size == 0:
                return np.empty(
                    (0, arr.shape[1] if arr.ndim > 1 else 1), dtype=arr.dtype
                )
            return arr[indices]

        pos_pairs = gather(self.positive_pairs, pos_idx)
        neg_pairs = gather(self.negative_pairs, neg_idx)
        hop_pairs = gather(self.hop_pairs, hop_idx)
        pos_flow = (
            self.positive_flow[pos_idx]
            if pos_idx.size
            else np.empty((0,), dtype=np.float32)
        )
        hop_vals = (
            self.hop_values[hop_idx]
            if hop_idx.size
            else np.empty((0,), dtype=np.float32)
        )

        return {
            "positive_pairs": torch.from_numpy(pos_pairs.T).long()
            if pos_pairs.size
            else torch.empty((2, 0), dtype=torch.long),
            "positive_flow": torch.from_numpy(pos_flow).float()
            if pos_flow.size
            else torch.empty((0,), dtype=torch.float),
            "negative_pairs": torch.from_numpy(neg_pairs.T).long()
            if neg_pairs.size
            else torch.empty((2, 0), dtype=torch.long),
            "hop_pairs": torch.from_numpy(hop_pairs.T).long()
            if hop_pairs.size
            else torch.empty((2, 0), dtype=torch.long),
            "hop_values": torch.from_numpy(hop_vals).float()
            if hop_vals.size
            else torch.empty((0,), dtype=torch.float),
        }

    def _sample_indices(self, length: int, count: int) -> np.ndarray:
        if length == 0 or count <= 0:
            return np.empty((0,), dtype=np.int64)
        replace = length < count
        return self.rng.choice(length, size=count, replace=replace)


# ---------------------------------------------------------------------------
# Region embedder trainer
# ---------------------------------------------------------------------------


class RegionEmbedderTrainer:
    def __init__(self, config: RegionTrainerConfig) -> None:
        self.config = config
        self.device = self._select_device(config.training.device)
        if config.training.seed is not None:
            torch.manual_seed(config.training.seed)
            np.random.seed(config.training.seed)

        # Create dataset
        from data.region_graph_dataset import RegionGraphDataset

        self.dataset = RegionGraphDataset(
            zarr_path=config.data.zarr_path,
            normalize_features=config.data.normalize_features,
            device=self.device,
        )

        # Load required data
        self.features = self.dataset.get_all_features()
        self.edge_index = self.dataset.get_edge_index()
        raw_flow_matrix = self.dataset.get_flow_matrix()
        self.flow_source = self.dataset.flow_source or (
            "mobility" if raw_flow_matrix is not None else "adjacency"
        )
        self.flow_matrix = raw_flow_matrix if self.flow_source == "mobility" else None
        self.region_ids = self.dataset.get_region_ids()

        self.num_nodes = self.features.size(0)
        self.feature_dim = self.features.size(1)

        # Validate dimensions
        if self.edge_index.numel() == 0:
            raise ValueError(
                "Edge index is empty; provide at least one edge for region training."
            )
        if self.edge_index.max().item() >= self.num_nodes:
            raise ValueError(
                "Edge index contains node indices outside the feature matrix range."
            )
        if self.flow_matrix is not None and self.flow_matrix.size(0) != self.num_nodes:
            raise ValueError(
                "Flow matrix must align with the number of regions in the feature tensor."
            )
        if len(self.region_ids) != self.num_nodes:
            raise ValueError("Number of region IDs does not match number of regions.")

        self.encoder = InductiveNodeEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.config.encoder.hidden_dim,
            output_dim=self.config.encoder.embedding_dim,
            num_layers=self.config.encoder.num_layers,
            aggregation=self.config.encoder.aggregation,
            dropout=self.config.encoder.dropout,
            residual=self.config.encoder.residual,
            normalize=self.config.encoder.normalize,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        self.spatial_prior = SpatialContiguityPrior(
            max_hops=self.config.sampling.max_hops, cache_distances=True
        )
        self.hop_distances = self.spatial_prior.compute_hop_distances(
            self.edge_index.cpu(), num_nodes=self.num_nodes
        ).to(self.device)

        self.primary_loss = self._build_primary_loss()
        rng = np.random.default_rng(self.config.training.seed)
        self.pair_sampler = PairSampler(
            self.flow_matrix,
            self.edge_index,
            self.hop_distances,
            self.config.sampling,
            rng,
        )
        self.best_loss = float("inf")
        self.best_state: dict[str, Any] | None = None
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        epochs = self.config.training.epochs
        for epoch in range(1, epochs + 1):
            metrics = self._train_one_epoch(epoch)
            self.history.append(metrics)
            if metrics["total_loss"] < self.best_loss:
                self.best_loss = metrics["total_loss"]
                self.best_state = {
                    k: v.detach().cpu() for k, v in self.encoder.state_dict().items()
                }

        if self.best_state is not None:
            self.encoder.load_state_dict(self.best_state)

        embeddings = None
        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder(self.features, self.edge_index).cpu()

        clusters = (
            self._cluster_embeddings(embeddings)
            if self.config.clustering.enabled
            else None
        )

        artifacts = self._save_artifacts(embeddings, clusters)

        return {
            "best_loss": float(self.best_loss),
            "epochs": epochs,
            "artifacts": artifacts,
            "cluster_labels": clusters,
            "region_ids": self.region_ids,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "edges": int(self.edge_index.size(1)),
            "flow_matrix": "available" if self.flow_matrix is not None else "missing",
            "embedding_dim": self.config.encoder.embedding_dim,
            "device": str(self.device),
            "epochs": self.config.training.epochs,
        }

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.encoder.train()
        self.optimizer.zero_grad()

        embeddings = self.encoder(self.features, self.edge_index)
        loss_outputs = self._compute_primary_loss(embeddings)
        total_loss = loss_outputs["total_loss"]

        pair_losses = self._compute_region2vec_losses(embeddings)
        total_loss = (
            total_loss
            + self.config.loss.ratio_weight * pair_losses["ratio_loss"]
            + self.config.loss.hop_weight * pair_losses["hop_loss"]
        )

        total_loss.backward()
        if self.config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.config.training.gradient_clip
            )
        self.optimizer.step()

        metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "base_loss": float(loss_outputs["base_loss"].detach().cpu()),
            "ratio_loss": float(pair_losses["ratio_loss"].detach().cpu()),
            "hop_loss": float(pair_losses["hop_loss"].detach().cpu()),
        }

        if epoch % self.config.training.log_every == 0 or epoch == 1:
            logger.info(
                "Epoch %d | loss=%.4f (base=%.4f, ratio=%.4f, hop=%.4f)",
                epoch,
                metrics["total_loss"],
                metrics["base_loss"],
                metrics["ratio_loss"],
                metrics["hop_loss"],
            )

        return metrics

    def _compute_primary_loss(
        self, embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.primary_loss, (CommunityOrientedLoss, SpatialOnlyLoss)):
            flow = (
                self.flow_matrix
                if self.flow_matrix is not None
                else torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
            )
            loss_dict = self.primary_loss(embeddings, flow, self.edge_index)
            base_loss = loss_dict.get(
                "total_loss", torch.tensor(0.0, device=self.device)
            )
        else:
            base_loss = torch.tensor(0.0, device=self.device)
        return {
            "total_loss": base_loss,
            "base_loss": base_loss,
        }

    def _compute_region2vec_losses(
        self, embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        samples = self.pair_sampler.sample()
        eps = 1e-8
        ratio_loss = torch.tensor(0.0, device=self.device)
        hop_loss = torch.tensor(0.0, device=self.device)

        pos_pairs = samples["positive_pairs"].to(self.device)
        neg_pairs = samples["negative_pairs"].to(self.device)
        if pos_pairs.numel() and neg_pairs.numel():
            pos_flow = samples["positive_flow"].to(self.device)
            d_pos = self._pairwise_distance(embeddings, pos_pairs)
            d_neg = self._pairwise_distance(embeddings, neg_pairs)
            ratio_loss = (torch.log(pos_flow + eps) * d_pos).mean() / (
                d_neg.mean() + eps
            )

        hop_pairs = samples["hop_pairs"].to(self.device)
        hop_values = samples["hop_values"].to(self.device)
        if hop_pairs.numel():
            d_hop = self._pairwise_distance(embeddings, hop_pairs)
            hop_loss = (
                d_hop / torch.clamp(torch.log(hop_values + 1.0), min=1e-4)
            ).mean()

        return {"ratio_loss": ratio_loss, "hop_loss": hop_loss}

    @staticmethod
    def _pairwise_distance(
        embeddings: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        i, j = pairs
        return torch.norm(embeddings[i] - embeddings[j], dim=1)

    # ------------------------------------------------------------------
    # Loss / clustering / artifacts
    # ------------------------------------------------------------------
    def _build_primary_loss(self) -> nn.Module:
        if self.config.loss.loss_type == "community":
            if self.flow_matrix is None:
                raise ValueError(
                    "Community loss requires a flow matrix. Provide `flow_path` in the config data section."
                )
            return CommunityOrientedLoss(
                temperature=self.config.loss.temperature,
                margin=self.config.loss.margin,
                spatial_weight=self.config.loss.spatial_weight,
                hop_threshold=self.config.sampling.hop_threshold,
                max_hops=self.config.sampling.max_hops,
                min_flow_threshold=self.config.sampling.min_flow_threshold,
                autocorrelation_weight=self.config.loss.autocorr_weight,
            )
        return SpatialOnlyLoss(
            temperature=self.config.loss.temperature,
            margin=self.config.loss.margin,
            hop_threshold=self.config.sampling.hop_threshold,
            max_hops=self.config.sampling.max_hops,
        )

    def _cluster_embeddings(self, embeddings: torch.Tensor) -> list[int] | None:
        n_clusters = min(self.config.clustering.num_clusters, embeddings.size(0))
        if n_clusters < 2:
            return None

        connectivity = None
        if self.config.clustering.compute_connectivity:
            src = self.edge_index[0].cpu().numpy()
            dst = self.edge_index[1].cpu().numpy()
            data = np.ones_like(src, dtype=np.float32)
            connectivity = sparse.csr_matrix(
                (data, (src, dst)), shape=(self.num_nodes, self.num_nodes)
            )
            connectivity = connectivity.maximum(connectivity.T)
            try:
                num_components, _ = sparse.csgraph.connected_components(connectivity)
                if num_components > 1:
                    logger.warning(
                        "Clustering connectivity graph has %d disconnected component(s);"
                        " Ward linkage will operate independently per island.",
                        int(num_components),
                    )
            except Exception as exc:  # pragma: no cover - scipy availability
                logger.warning("Failed to compute connectivity components: %s", exc)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config.clustering.linkage,
            connectivity=connectivity,
        )
        labels = model.fit_predict(embeddings.cpu().numpy())
        return labels.tolist()

    def _save_artifacts(
        self, embeddings: torch.Tensor, clusters: list[int] | None
    ) -> dict[str, str | None]:
        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        embedding_path = output_dir / self.config.output.embedding_filename
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in self.encoder.state_dict().items()
        }
        payload = {
            "embeddings": embeddings,
            "region_ids": self.region_ids,
            "config": self.config.to_dict(),
            "encoder_state_dict": state_dict,
            "feature_dim": self.feature_dim,
        }
        torch.save(payload, embedding_path)

        if self.config.output.save_numpy:
            np.save(embedding_path.with_suffix(".npy"), embeddings.cpu().numpy())

        metrics_path = output_dir / self.config.output.metrics_filename
        metrics_data = {
            "history": self.history,
            "best_loss": self.best_loss,
        }
        metrics_path.write_text(json.dumps(metrics_data, indent=2))

        cluster_path = None
        if clusters is not None:
            cluster_path = output_dir / self.config.output.cluster_labels_filename
            cluster_payload = {
                rid: label
                for rid, label in zip(self.region_ids, clusters, strict=False)
            }
            cluster_path.write_text(json.dumps(cluster_payload, indent=2))

        return {
            "embedding_path": str(embedding_path),
            "metrics_path": str(metrics_path),
            "cluster_labels_path": str(cluster_path) if cluster_path else None,
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
