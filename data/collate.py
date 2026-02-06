from __future__ import annotations

from typing import Any

from data.epi_dataset import EpiDatasetItem, collate_epiforecaster_batch


def collate_epidataset_batch(
    batch: list[EpiDatasetItem], *, require_region_index: bool = True
) -> dict[str, Any]:
    """Compatibility wrapper delegating to the shared EpiForecaster collate."""
    return collate_epiforecaster_batch(
        batch, require_region_index=require_region_index
    )

