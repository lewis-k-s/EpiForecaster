from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import evaluation.loaders as loaders


class _DummyDataset:
    def __init__(self, target_nodes: list[int]) -> None:
        self.target_nodes = target_nodes
        self.biomarker_preprocessor = object()
        self.mobility_preprocessor = object()
        self.preloaded_mobility = torch.ones(1, 1, 1)
        self.mobility_mask = torch.ones(1, 1, 1, dtype=torch.bool)
        self.region_embeddings = None

    def __len__(self) -> int:
        return len(self.target_nodes)


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            val_workers=0,
            test_workers=0,
            batch_size=4,
            pin_memory=False,
            prefetch_factor=None,
        ),
        model=SimpleNamespace(type=SimpleNamespace(regions=False)),
    )


def test_build_loader_from_config_full_uses_all_target_nodes(monkeypatch) -> None:
    train = _DummyDataset([0, 1])
    val = _DummyDataset([2])
    test = _DummyDataset([3])
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        loaders,
        "build_datasets",
        lambda config: SimpleNamespace(
            train=train,
            val=val,
            test=test,
            real_run_id=None,
            region_embedding_store=None,
        ),
    )

    class _FakeDataset:
        def __init__(self, **kwargs):
            captured["dataset_kwargs"] = kwargs
            self.region_embeddings = None

        def __len__(self) -> int:
            return 4

    monkeypatch.setattr(loaders, "EpiDataset", _FakeDataset)
    config = _make_config()

    loader, region_embeddings = loaders.build_loader_from_config(
        config,
        split="full",
        device="cpu",
    )

    assert region_embeddings is None
    assert loader.batch_size == 4
    dataset_kwargs = captured["dataset_kwargs"]
    assert dataset_kwargs["target_nodes"] == [0, 1, 2, 3]
    assert dataset_kwargs["context_nodes"] == [0, 1, 2, 3]
    assert dataset_kwargs["biomarker_preprocessor"] is val.biomarker_preprocessor
    assert dataset_kwargs["mobility_preprocessor"] is val.mobility_preprocessor


def test_build_loader_from_config_full_rejects_curriculum(monkeypatch) -> None:
    monkeypatch.setattr(
        loaders,
        "build_datasets",
        lambda config: SimpleNamespace(
            train=torch.utils.data.ConcatDataset([_DummyDataset([99])]),
            val=_DummyDataset([0]),
            test=_DummyDataset([1]),
            real_run_id="real",
            region_embedding_store=None,
        ),
    )

    with pytest.raises(ValueError, match="split='full' is not supported"):
        loaders.build_loader_from_config(_make_config(), split="full", device="cpu")
