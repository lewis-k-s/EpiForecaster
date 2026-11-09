"""Within-window information between inferred latents, observations and forecasts.

Ground-truth-free indicators of whether the latent that feeds each observation
head tracks the observed series and drives the forecast. They complement
``TorchLatentPlausibilityAccumulator``: plausibility asks whether the latent
levels are epidemiologically sensible, these indicators ask whether the latent
carries information about the observed dynamics.

Every series is centred within its forecast window over the supervised days
(weight > 0), so a region's level and the delta-forecasting anchor do not
count as information. The latent is compared in ``log1p(latent * 1e5)``, the
per-100k scale of the clinical targets, which is monotone in the latent.

Metrics, per head ``<head>`` in hosp, ww, cases, deaths (prefix
``latent_info_<head>_``):

* ``r2_lat_obs``, ``r2_lat_pred``: squared Pearson correlation on the same day.
* ``mi_lat_obs``, ``mi_lat_pred``, ``mi_pred_obs``: Gaussian-copula mutual
  information in bits (rank based, average ranks for ties).
* ``te_lat_obs``, ``te_pred_obs``: linear-Gaussian transfer entropy in bits
  from the source on day ``h`` to the observation on day ``h + step`` given
  the observation on day ``h``; ``step`` is 1 day for cases and deaths and
  7 days for hospitalizations and wastewater (their reporting cadence).
* ``sign_lat_obs``, ``sign_lat_pred``: fraction of ``step``-day changes with
  the same sign, over pairs where both changes are nonzero.
* ``resid_share``: within-window variance of the residual contribution over
  that of the forecast location. It can exceed 1 when the residual opposes
  the mechanistic path.
* ``r2_lat_obs_weekly``, ``resid_share_weekly`` (cases and deaths only): the
  same quantities on 7-day block means of the centred series (horizon days
  1-7, 8-14, ...). Daily reporting has a day-of-week cycle that the latent
  cannot produce and the residual carries; the block means remove it.
* ``lag_r2_lat_obs_<m|p><days>``: ``r2`` of the latent on day ``h`` against
  the observation on day ``h + lag``; ``p`` lags mean the latent leads.
  Lags span +-14 days in 1-day steps for cases and deaths and +-21 days in
  7-day steps for hospitalizations and wastewater. ``best_lag_lat_obs`` and
  ``best_lag_r2_lat_obs`` give the maximising lag and its ``r2``.
* ``<metric>_null95`` for the four latent-observation metrics: the 95th
  percentile over ``RESAMPLE_COUNT`` permutations that pair each window's
  observations with another window's latent trajectory.
* ``<metric>_q05`` / ``<metric>_q95``: 5th and 95th percentiles over
  ``RESAMPLE_COUNT`` bootstrap resamples of whole regions. They cover region
  sampling only, not training randomness.
* ``window_count``, ``node_count``, ``observation_count`` and
  ``resample_window_count`` (windows entering permutations and bootstraps; a
  seeded subsample of at most ``MAX_RESAMPLE_WINDOWS``).

Overlapping windows (stride below the horizon) make the effective sample
smaller than the observation count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LATENT_INFORMATION_PREFIX = "latent_info"
LATENT_INFORMATION_HEADS = ("hosp", "ww", "cases", "deaths")
STEP_DAYS = {"hosp": 7, "ww": 7, "cases": 1, "deaths": 1}
MAX_LAG_DAYS = {"hosp": 21, "ww": 21, "cases": 14, "deaths": 14}
RESAMPLE_COUNT = 200
MAX_RESAMPLE_WINDOWS = 20_000
RESAMPLE_SEED = 0
_NULL_METRICS = ("r2_lat_obs", "mi_lat_obs", "te_lat_obs", "sign_lat_obs")
_FIELDS = ("latent", "observed", "prediction", "residual", "weights")
_NAN = float("nan")


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks starting at 1 with ties sharing their average rank."""
    _, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True
    )
    ends = torch.cumsum(counts, dim=0).to(torch.float64)
    average = ends - (counts.to(torch.float64) - 1.0) / 2.0
    return average[inverse]


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return _NAN
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = torch.sqrt((xc * xc).sum() * (yc * yc).sum())
    if float(denominator) <= 0.0:
        return 0.0
    return float((xc * yc).sum() / denominator)


def _r2(x: torch.Tensor, y: torch.Tensor) -> float:
    rho = _correlation(x, y)
    return rho * rho if math.isfinite(rho) else _NAN


def _variance_share(part: torch.Tensor, whole: torch.Tensor) -> float:
    if whole.numel() < 2:
        return _NAN
    whole_variance = float(whole.var())
    return float(part.var()) / whole_variance if whole_variance > 0 else _NAN


def _bits_from_correlation(rho: float) -> float:
    if not math.isfinite(rho):
        return _NAN
    return -0.5 * math.log2(max(1.0 - rho * rho, 1e-300))


def _gaussian_copula_mi(x: torch.Tensor, y: torch.Tensor) -> float:
    """Gaussian-copula mutual information in bits, a lower bound."""
    if x.numel() < 3:
        return _NAN
    count = x.numel() + 1
    gx = torch.special.ndtri(_average_ranks(x) / count)
    gy = torch.special.ndtri(_average_ranks(y) / count)
    return _bits_from_correlation(_correlation(gx, gy))


def _residualize(target: torch.Tensor, regressor: torch.Tensor) -> torch.Tensor:
    """Residual of ``target`` after a least-squares fit on ``[1, regressor]``."""
    rc = regressor - regressor.mean()
    tc = target - target.mean()
    denominator = (rc * rc).sum()
    slope = (rc * tc).sum() / denominator if float(denominator) > 0 else 0.0
    return tc - slope * rc


def _gaussian_transfer_entropy(
    source: torch.Tensor, destination_next: torch.Tensor, destination: torch.Tensor
) -> float:
    """Linear-Gaussian transfer entropy source -> destination_next | destination."""
    if source.numel() < 4:
        return _NAN
    return _bits_from_correlation(
        _correlation(
            _residualize(source, destination),
            _residualize(destination_next, destination),
        )
    )


def _sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    keep = (a != 0) & (b != 0)
    if not bool(keep.any()):
        return _NAN
    return float((torch.sign(a[keep]) == torch.sign(b[keep])).double().mean())


def _centre(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Subtract each window's mean over its valid days; invalid days become 0."""
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    filled = torch.where(valid, values, torch.zeros_like(values))
    mean = filled.sum(dim=1, keepdim=True) / counts
    return torch.where(valid, values - mean, torch.zeros_like(values))


def _weekly_block_means(
    values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Means over 7-day blocks of valid days and the blocks with any valid day."""
    horizon = values.shape[1]
    blocks = math.ceil(horizon / 7)
    pad = blocks * 7 - horizon
    filled = F.pad(torch.where(valid, values, torch.zeros_like(values)), (0, pad))
    counts = F.pad(valid.to(values.dtype), (0, pad))
    sums = filled.reshape(-1, blocks, 7).sum(dim=2)
    days = counts.reshape(-1, blocks, 7).sum(dim=2)
    return sums / days.clamp_min(1.0), days > 0


def _lag_pairs(
    source: torch.Tensor, destination: torch.Tensor, valid: torch.Tensor, lag: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source on day h with destination on day h + lag, both valid."""
    horizon = valid.shape[1]
    if abs(lag) >= horizon:
        empty = source.new_empty(0)
        return empty, empty
    if lag >= 0:
        src, dst = source[:, : horizon - lag], destination[:, lag:]
        keep = valid[:, : horizon - lag] & valid[:, lag:]
    else:
        src, dst = source[:, -lag:], destination[:, : horizon + lag]
        keep = valid[:, -lag:] & valid[:, : horizon + lag]
    return src[keep], dst[keep]


def _lag_label(lag: int) -> str:
    return f"p{lag}" if lag >= 0 else f"m{-lag}"


class _HeadSeries:
    """Centred within-window series of one head."""

    def __init__(
        self,
        *,
        latent: torch.Tensor,
        observed: torch.Tensor,
        prediction: torch.Tensor,
        residual: torch.Tensor,
        valid: torch.Tensor,
        step: int,
    ) -> None:
        self.latent_raw = latent
        self.valid = valid
        self.step = step
        self.latent = _centre(latent, valid)
        self.observed = _centre(observed, valid)
        self.prediction = _centre(prediction, valid)
        self.residual = _centre(residual, valid)

    def _copy(self, **fields: torch.Tensor) -> _HeadSeries:
        clone = object.__new__(_HeadSeries)
        clone.__dict__.update(self.__dict__)
        clone.__dict__.update(fields)
        return clone

    def with_latent(self, latent: torch.Tensor) -> _HeadSeries:
        """Copy with a different raw latent, centred on this series' valid days."""
        return self._copy(latent_raw=latent, latent=_centre(latent, self.valid))

    def subset(self, rows: torch.Tensor) -> _HeadSeries:
        return self._copy(
            **{
                name: getattr(self, name)[rows]
                for name in (
                    "latent_raw",
                    "valid",
                    "latent",
                    "observed",
                    "prediction",
                    "residual",
                )
            }
        )

    def _changes(self, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``step``-day changes of the raw latent and of ``other``, both valid."""
        step = self.step
        keep = self.valid[:, :-step] & self.valid[:, step:]
        latent_change = self.latent_raw[:, step:] - self.latent_raw[:, :-step]
        other_change = other[:, step:] - other[:, :-step]
        return latent_change[keep], other_change[keep]

    def latent_observation_metrics(self) -> dict[str, float]:
        valid = self.valid
        latent = self.latent[valid]
        observed = self.observed[valid]
        lat_now, obs_next = _lag_pairs(self.latent, self.observed, valid, self.step)
        obs_now, _ = _lag_pairs(self.observed, self.observed, valid, self.step)
        lat_change, obs_change = self._changes(self.observed)
        return {
            "r2_lat_obs": _r2(latent, observed),
            "mi_lat_obs": _gaussian_copula_mi(latent, observed),
            "te_lat_obs": _gaussian_transfer_entropy(lat_now, obs_next, obs_now),
            "sign_lat_obs": _sign_agreement(lat_change, obs_change),
        }

    def metrics(self) -> dict[str, float]:
        valid = self.valid
        latent = self.latent[valid]
        observed = self.observed[valid]
        prediction = self.prediction[valid]
        pred_now, obs_next = _lag_pairs(
            self.prediction, self.observed, valid, self.step
        )
        obs_now, _ = _lag_pairs(self.observed, self.observed, valid, self.step)
        lat_change, pred_change = self._changes(self.prediction)
        metrics = {
            **self.latent_observation_metrics(),
            "r2_lat_pred": _r2(latent, prediction),
            "mi_lat_pred": _gaussian_copula_mi(latent, prediction),
            "sign_lat_pred": _sign_agreement(lat_change, pred_change),
            "mi_pred_obs": _gaussian_copula_mi(prediction, observed),
            "te_pred_obs": _gaussian_transfer_entropy(pred_now, obs_next, obs_now),
            "resid_share": _variance_share(self.residual[valid], prediction),
        }
        if self.step == 1:
            weekly = {
                name: _weekly_block_means(getattr(self, name), valid)
                for name in ("latent", "observed", "prediction", "residual")
            }
            blocks = weekly["latent"][1]
            metrics["r2_lat_obs_weekly"] = _r2(
                weekly["latent"][0][blocks], weekly["observed"][0][blocks]
            )
            metrics["resid_share_weekly"] = _variance_share(
                weekly["residual"][0][blocks], weekly["prediction"][0][blocks]
            )
        return metrics

    def lag_metrics(self, max_lag: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        best_lag, best_r2 = _NAN, -1.0
        for lag in range(-max_lag, max_lag + 1, self.step):
            source, destination = _lag_pairs(
                self.latent, self.observed, self.valid, lag
            )
            value = _r2(source, destination)
            metrics[f"lag_r2_lat_obs_{_lag_label(lag)}"] = value
            if math.isfinite(value) and value > best_r2:
                best_lag, best_r2 = float(lag), value
        metrics["best_lag_lat_obs"] = best_lag
        metrics["best_lag_r2_lat_obs"] = best_r2 if best_r2 >= 0 else _NAN
        return metrics


class TorchLatentInformationAccumulator:
    """Collect per-window head series and compute latent-information indicators.

    ``update`` moves each batch to CPU float64, so enable this accumulator for
    final evaluations rather than per-epoch validation.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, dict[str, list[torch.Tensor]]] = {
            head: {field: [] for field in (*_FIELDS, "nodes")}
            for head in LATENT_INFORMATION_HEADS
        }

    def update(
        self,
        *,
        head: str,
        latent: torch.Tensor,
        observed: torch.Tensor,
        prediction: torch.Tensor,
        residual: torch.Tensor,
        weights: torch.Tensor,
        nodes: torch.Tensor,
    ) -> None:
        """Add one batch of ``[B, H]`` series for ``head``; ``nodes`` is ``[B]``.

        ``latent`` is the population-fraction latent that feeds the head; it
        is compared as ``log1p(latent * 1e5)``.
        """
        if head not in self.chunks:
            raise ValueError(f"Unknown latent-information head {head!r}")
        series = {
            "latent": latent,
            "observed": observed,
            "prediction": prediction,
            "residual": residual,
            "weights": weights,
        }
        shapes = {tuple(value.shape) for value in series.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(
                f"Latent-information series for {head!r} must share one [B, H] "
                f"shape, got {sorted(shapes)}"
            )
        if tuple(nodes.shape) != (latent.shape[0],):
            raise ValueError(
                f"nodes must have shape [{latent.shape[0]}], got {tuple(nodes.shape)}"
            )
        target = self.chunks[head]
        for name, value in series.items():
            # Copy first, then cast: MPS tensors cannot hold float64.
            target[name].append(value.detach().to("cpu").to(torch.float64))
        target["nodes"].append(nodes.detach().to("cpu").to(torch.long))

    def _series(self, head: str) -> tuple[_HeadSeries, torch.Tensor] | None:
        chunks = self.chunks[head]
        if not chunks["latent"]:
            return None
        data = {name: torch.cat(values) for name, values in chunks.items()}
        latent = torch.log1p(data["latent"].clamp_min(0.0) * 1e5)
        valid = (
            (data["weights"] > 0)
            & torch.isfinite(data["weights"])
            & torch.isfinite(latent)
            & torch.isfinite(data["observed"])
            & torch.isfinite(data["prediction"])
            & torch.isfinite(data["residual"])
        )
        # A window needs two supervised days to have a within-window deviation.
        rows = valid.sum(dim=1) >= 2
        series = _HeadSeries(
            latent=latent[rows],
            observed=data["observed"][rows],
            prediction=data["prediction"][rows],
            residual=data["residual"][rows],
            valid=valid[rows],
            step=STEP_DAYS[head],
        )
        return series, data["nodes"][rows]

    def finalize(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for head in LATENT_INFORMATION_HEADS:
            prefix = f"{LATENT_INFORMATION_PREFIX}_{head}"
            resolved = self._series(head)
            window_count = 0 if resolved is None else int(resolved[1].numel())
            metrics[f"{prefix}_window_count"] = window_count
            if resolved is None or window_count == 0:
                continue
            series, nodes = resolved
            metrics[f"{prefix}_node_count"] = int(torch.unique(nodes).numel())
            metrics[f"{prefix}_observation_count"] = int(series.valid.sum())
            point = series.metrics()
            metrics.update({f"{prefix}_{k}": v for k, v in point.items()})
            metrics.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in series.lag_metrics(MAX_LAG_DAYS[head]).items()
                }
            )
            metrics.update(_resampled(prefix, series, nodes, point))
        return metrics


def _resampled(
    prefix: str,
    series: _HeadSeries,
    nodes: torch.Tensor,
    point: dict[str, float],
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(RESAMPLE_SEED)
    window_count = int(nodes.numel())
    if window_count > MAX_RESAMPLE_WINDOWS:
        rows = (
            torch.randperm(window_count, generator=generator)[:MAX_RESAMPLE_WINDOWS]
            .sort()
            .values
        )
        series, nodes = series.subset(rows), nodes[rows]
        window_count = MAX_RESAMPLE_WINDOWS
    results: dict[str, float | int] = {f"{prefix}_resample_window_count": window_count}

    null_draws: dict[str, list[float]] = {name: [] for name in _NULL_METRICS}
    for _ in range(RESAMPLE_COUNT):
        donors = torch.randperm(window_count, generator=generator)
        draw = series.with_latent(
            series.latent_raw[donors]
        ).latent_observation_metrics()
        for name in _NULL_METRICS:
            null_draws[name].append(draw[name])
    for name, values in null_draws.items():
        results[f"{prefix}_{name}_null95"] = _percentile(values, 0.95)

    _, node_index = torch.unique(nodes, return_inverse=True)
    windows_by_node = [
        torch.nonzero(node_index == i).flatten()
        for i in range(int(node_index.max()) + 1)
    ]
    boot_draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(RESAMPLE_COUNT):
        picks = torch.randint(
            len(windows_by_node), (len(windows_by_node),), generator=generator
        )
        rows = torch.cat([windows_by_node[int(i)] for i in picks])
        draw = series.subset(rows).metrics()
        for name in point:
            boot_draws[name].append(draw[name])
    for name, values in boot_draws.items():
        results[f"{prefix}_{name}_q05"] = _percentile(values, 0.05)
        results[f"{prefix}_{name}_q95"] = _percentile(values, 0.95)
    return results


def _percentile(values: list[float], level: float) -> float:
    finite = torch.tensor([v for v in values if math.isfinite(v)], dtype=torch.float64)
    if finite.numel() == 0:
        return _NAN
    return float(torch.quantile(finite, level))
