"""
Time series forecaster trainer for cases-only baselines.

This module provides a lightweight trainer that consumes the COVID cases loader
and trains a simple recurrent model without geospatial inputs. It is intended
for rapid experimentation with ablated or simplified variants of the full
pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from data.cases_loader import create_cases_loader

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesTrainerConfig:
    data_dir: str
    cases_file: str
    cases_normalization: str
    min_cases_threshold: int
    cases_fill_missing: str
    forecast_horizon: int
    context_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    num_layers: int
    dropout: float
    device: str
    output_dir: str
    save_model: bool
    seed: int | None
    train_split: float
    val_split: float
    test_split: float
    start_date: str | None
    end_date: str | None


class CasesSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding window dataset over cases tensor."""

    def __init__(
        self, cases_tensor: Tensor, context_length: int, forecast_horizon: int
    ) -> None:
        if cases_tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor [num_series, time], got {cases_tensor.shape}"
            )
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        num_series, num_timepoints = cases_tensor.shape
        min_required = context_length + forecast_horizon
        if num_timepoints < min_required:
            raise ValueError(
                f"Not enough timepoints ({num_timepoints}) for context={context_length} "
                f"and horizon={forecast_horizon}"
            )

        self.cases_tensor = cases_tensor.float()
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self._samples: list[tuple[int, int]] = []

        max_start = num_timepoints - min_required + 1
        for series_idx in range(num_series):
            for start_idx in range(max_start):
                self._samples.append((series_idx, start_idx))

        if not self._samples:
            raise ValueError("Dataset construction produced zero samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        series_idx, start_idx = self._samples[index]
        ctx_end = start_idx + self.context_length
        tgt_end = ctx_end + self.forecast_horizon

        context = self.cases_tensor[series_idx, start_idx:ctx_end]
        target = self.cases_tensor[series_idx, ctx_end:tgt_end]
        return context, target


class TimeSeriesForecaster(nn.Module):
    """Simple GRU-based forecaster for univariate sequences."""

    def __init__(
        self, hidden_dim: int, num_layers: int, forecast_horizon: int, dropout: float
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, context: Tensor) -> Tensor:
        # context: [batch, seq_len]
        seq = context.unsqueeze(-1)  # [batch, seq_len, 1]
        _, hidden = self.gru(seq)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subset_tensor_by_date(
    cases_tensor: Tensor,
    date_range: tuple[datetime, datetime],
    start_date: str | None,
    end_date: str | None,
) -> tuple[Tensor, pd.DatetimeIndex]:
    if start_date is None and end_date is None:
        time_index = pd.date_range(date_range[0], date_range[1], freq="D")
        return cases_tensor, time_index

    time_index = pd.date_range(date_range[0], date_range[1], freq="D")
    mask = pd.Series(True, index=time_index)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        mask &= time_index >= start_ts
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        mask &= time_index <= end_ts

    filtered = cases_tensor[:, mask.values]
    return filtered, time_index[mask]


def _prepare_output_dirs(base_output_dir: str) -> tuple[Path, Path]:
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = base_path / f"run_{timestamp}"
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = base_path / "latest"
    if latest_dir.exists():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir, latest_dir


def _compute_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = math.sqrt(mse)
    return {"mae": mae, "rmse": rmse, "mse": mse}


class TimeSeriesTrainer:
    """Trainer entry-point for cases-only forecasting."""

    def __init__(self, config: TimeSeriesTrainerConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.timestamped_dir: Path | None = None
        self.latest_dir: Path | None = None

    @classmethod
    def from_args(cls, args: Any) -> "TimeSeriesTrainer":
        config = TimeSeriesTrainerConfig(
            data_dir=args.data_dir,
            cases_file=args.cases_file,
            cases_normalization=getattr(args, "cases_normalization", "log1p"),
            min_cases_threshold=getattr(args, "min_cases_threshold", 0),
            cases_fill_missing=getattr(args, "cases_fill_missing", "forward_fill"),
            forecast_horizon=args.forecast_horizon,
            context_length=getattr(args, "context_length", 28),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
            output_dir=args.output_dir,
            save_model=getattr(args, "save_model", False),
            seed=getattr(args, "seed", None),
            train_split=getattr(args, "train_split", 0.7),
            val_split=getattr(args, "val_split", 0.15),
            test_split=getattr(args, "test_split", 0.15),
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
        )
        return cls(config)

    def run(self) -> dict[str, Any]:
        logger.info("Starting cases-only time series training")
        _set_seed(self.config.seed)

        cases_loader = create_cases_loader(
            cases_file=self._resolve_cases_path(),
            normalization=self.config.cases_normalization,
            min_cases_threshold=self.config.min_cases_threshold,
            fill_missing=self.config.cases_fill_missing,
        )

        tensor, time_index = _subset_tensor_by_date(
            cases_loader.cases_tensor,
            cases_loader.date_range,
            self.config.start_date,
            self.config.end_date,
        )
        dataset = CasesSequenceDataset(
            tensor, self.config.context_length, self.config.forecast_horizon
        )

        splits = self._compute_splits(len(dataset))
        train_set, val_set, test_set = random_split(
            dataset,
            splits,
            generator=torch.Generator().manual_seed(self.config.seed or 0),
        )

        train_loader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False
        )

        model = TimeSeriesForecaster(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            forecast_horizon=self.config.forecast_horizon,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.timestamped_dir, self.latest_dir = _prepare_output_dirs(
            self.config.output_dir
        )
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_loss = self._evaluate_loss(model, val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                f"[Timeseries] Epoch {epoch}/{self.config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

        metrics = self._evaluate_metrics(model, test_loader)
        results = {
            "variant": "cases_timeseries",
            "config": asdict(self.config),
            "history": history,
            "metrics": metrics,
            "time_index_start": time_index[0].isoformat(),
            "time_index_end": time_index[-1].isoformat(),
            "output_dir": str(self.timestamped_dir),
        }

        self._save_artifacts(model, results)
        return results

    def _resolve_device(self, device_choice: str) -> torch.device:
        if device_choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_choice)

    def _resolve_cases_path(self) -> str:
        cases_path = Path(self.config.cases_file)
        if cases_path.is_absolute():
            return str(cases_path)
        return str(Path(self.config.data_dir) / cases_path)

    def _compute_splits(self, total_samples: int) -> list[int]:
        ratios = np.array(
            [self.config.train_split, self.config.val_split, self.config.test_split],
            dtype=float,
        )
        if np.allclose(ratios.sum(), 0.0):
            raise ValueError("At least one split ratio must be positive")
        ratios = ratios / ratios.sum()
        lengths = np.floor(ratios * total_samples).astype(int)
        remainder = total_samples - int(lengths.sum())
        idx = 0
        while remainder > 0:
            lengths[idx % len(lengths)] += 1
            remainder -= 1
            idx += 1

        # Ensure training split has at least one sample whenever possible
        if total_samples > 0 and lengths[0] == 0:
            for i in range(1, len(lengths)):
                if lengths[i] > 0:
                    lengths[i] -= 1
                    lengths[0] += 1
                    break
            if lengths[0] == 0:
                lengths[0] = 1

        total_adjustment = total_samples - int(lengths.sum())
        if total_adjustment != 0:
            lengths[-1] += total_adjustment

        lengths = np.clip(lengths, a_min=0, a_max=None)
        return lengths.astype(int).tolist()

    def _train_epoch(
        self,
        model: TimeSeriesForecaster,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        loss_fn = nn.MSELoss()

        for context, target in loader:
            context = context.to(self.device)
            target = target.to(self.device)
            optimizer.zero_grad()
            prediction = model(context)
            loss = loss_fn(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = context.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _evaluate_loss(
        self,
        model: TimeSeriesForecaster,
        loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> float:
        model.eval()
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for context, target in loader:
                context = context.to(self.device)
                target = target.to(self.device)
                prediction = model(context)
                loss = loss_fn(prediction, target)
                batch_size = context.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return total_loss / max(total_samples, 1)

    def _evaluate_metrics(
        self,
        model: TimeSeriesForecaster,
        loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> dict[str, float]:
        model.eval()
        preds: list[Tensor] = []
        targets: list[Tensor] = []
        with torch.no_grad():
            for context, target in loader:
                context = context.to(self.device)
                prediction = model(context)
                preds.append(prediction.cpu())
                targets.append(target)

        all_preds = torch.cat(preds, dim=0)
        all_targets = torch.cat(targets, dim=0)
        return _compute_metrics(all_preds, all_targets)

    def _save_artifacts(
        self, model: TimeSeriesForecaster, results: dict[str, Any]
    ) -> None:
        if self.timestamped_dir is None or self.latest_dir is None:
            raise RuntimeError("Output directories not prepared")

        results_path = self.timestamped_dir / "results.json"
        with results_path.open("w") as fh:
            json.dump(results, fh, indent=2)

        latest_results = self.latest_dir / "results.json"
        latest_results.write_text(results_path.read_text())

        if self.config.save_model:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(self.config)},
                self.timestamped_dir / "model_state.pt",
            )
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(self.config)},
                self.latest_dir / "model_state.pt",
            )
