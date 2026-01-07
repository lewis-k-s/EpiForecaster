from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class DataQualityThresholds:
    min_notna_fraction: float = 0.99
    min_std_epsilon: float = 1e-12


def _to_float(value) -> float:
    # Handle xarray scalars / numpy scalars / Python floats and dask-backed values.
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def validate_notna_and_std(
    data: xr.DataArray | xr.Dataset,
    *,
    name: str,
    var: str | None = None,
    thresholds: DataQualityThresholds = DataQualityThresholds(),
) -> None:
    """
    Validate that processed data is non-trivial.

    This is meant to catch silent preprocessing failures where a variable is
    fully missing or effectively constant (e.g. after a bad reindex).
    """
    if isinstance(data, xr.Dataset):
        if var is None:
            raise ValueError("`var` must be provided when validating an xr.Dataset")
        if var not in data:
            raise KeyError(f"Variable {var!r} not found in dataset for {name}")
        da = data[var]
    else:
        da = data

    total_count = int(da.size)
    if total_count == 0:
        raise ValueError(
            f"{name}: empty array after preprocessing (dims={dict(da.sizes)!r})"
        )

    notna_fraction = _to_float(da.notnull().mean())
    if not (notna_fraction == notna_fraction):
        raise ValueError(
            f"{name}: notna fraction is NaN (dims={dict(da.sizes)!r}); "
            "this usually indicates an empty reduction or all-missing data"
        )

    if notna_fraction < thresholds.min_notna_fraction:
        notna_count = int(_to_float(da.notnull().sum()))
        raise ValueError(
            f"{name}: notna fraction {notna_fraction:.6f} below "
            f"threshold {thresholds.min_notna_fraction:.6f} "
            f"({notna_count}/{total_count} non-missing; dims={dict(da.sizes)!r})"
        )

    std = _to_float(da.std())
    if not (std == std):
        raise ValueError(
            f"{name}: std is NaN (dims={dict(da.sizes)!r}); "
            "this usually indicates all-missing data"
        )

    if std <= thresholds.min_std_epsilon:
        raise ValueError(
            f"{name}: std {std:.6e} below/equal epsilon {thresholds.min_std_epsilon:.6e}"
        )
