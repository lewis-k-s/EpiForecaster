"""Model-to-loss integration test for the native path (review ruling R1).

Drives a real (tiny, seeded, CPU) ``NativeEpiForecaster`` forward on a
synthetic, hand-collated ``NativeEpiBatch`` — the DS-1 item-dict pattern the
dataset tests use, collated through ``collate_native_batch`` — and feeds the
RAW model outputs, including the ``[B, H + 1]`` cases nowcast width, to
``NativeJointInferenceLoss``. This is the "training remains connected" proof
(plan section 9) the pre-compaction review found missing: finite total,
finite per-head components, and finite gradients reaching the adapter, the
backbone, and every active head.
"""

from __future__ import annotations

import pytest
import torch

from data.native_batch import SOURCE_TYPE_IDS, NativeEpiBatch, collate_native_batch
from data.query_contract import ForecastQuery
from evaluation.losses import NativeJointInferenceLoss
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig

pytestmark = pytest.mark.epiforecaster

H = 14  # daily horizon -> W_max = ceil(14 / 7) = 2
L = 8
F_NATIVE = 3  # per-item native channels; collation adds the 5-channel one-hot
D_CTX = 2
K_COV = 2
SEED = 20260922

W_MAX = -(-H // 7)


def _query(
    source_type: str, source_id: str, origin: int, heads: set[str]
) -> ForecastQuery:
    return ForecastQuery(
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        forecast_origin=origin,
        horizon_days=H,
        requested_heads=frozenset(heads),
    )


def _item(
    *,
    query: ForecastQuery,
    head: str,
    source_pop: float,
    origin_dow: int,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    target_population_graph: float,
    target_stratum_pop_sum: torch.Tensor | None,
) -> dict:
    """One synthetic DS-1-shaped item; random fields come from the seeded
    global generator advanced by the caller."""
    return {
        "query": query,
        "head": head,
        "hist": torch.randn(L, F_NATIVE),
        "hist_mask": torch.ones(L, dtype=torch.bool),
        "ctx_feat": torch.randn(L, D_CTX),
        "ctx_coverage": 1.0,
        "temporal_covariates": torch.randn(L, K_COV),
        "source_type_id": SOURCE_TYPE_IDS[str(query.source_type)],
        "source_pop": source_pop,
        "origin_dow": origin_dow,
        "target": target,
        "target_mask": target_mask,
        "target_population_graph": target_population_graph,
        "target_stratum_pop_sum": target_stratum_pop_sum,
    }


def _build_model_and_batch() -> tuple[NativeEpiForecaster, NativeEpiBatch]:
    """Tiny seeded forecaster + one-row-per-head synthetic native batch.

    Real-data-style routing (plan section 11.1): municipality -> cases,
    abs -> hosp (Thursday origin, second Mon-Sun week partial), comarca ->
    deaths. Targets are raw counts per review ruling R2.
    """
    torch.manual_seed(SEED)
    cases_raw = 2.0 + 0.5 * torch.arange(H, dtype=torch.float32)
    cases_mask = torch.ones(H, dtype=torch.bool)
    cases_mask[3] = False  # one unobserved day: masked slots are weight-0
    items = [
        _item(
            query=_query("municipality", "08121", 10, {"cases"}),
            head="cases",
            source_pop=20_000.0,
            origin_dow=0,
            target=cases_raw,
            target_mask=cases_mask,
            target_population_graph=20_000.0,
            target_stratum_pop_sum=None,
        ),
        _item(
            query=_query("abs", "088", 11, {"hosp"}),
            head="hosp",
            source_pop=70_000.0,
            origin_dow=3,
            target=torch.tensor([12.0, 9.0]),
            target_mask=torch.tensor([True, False]),
            target_population_graph=70_000.0,
            target_stratum_pop_sum=torch.tensor([900.0, 900.0]),
        ),
        _item(
            query=_query("comarca", "06", 12, {"deaths"}),
            head="deaths",
            source_pop=180_000.0,
            origin_dow=4,
            target=0.2 + 0.1 * torch.arange(H, dtype=torch.float32),
            target_mask=torch.ones(H, dtype=torch.bool),
            target_population_graph=180_000.0,
            target_stratum_pop_sum=None,
        ),
    ]
    batch = collate_native_batch(items)
    config = NativeForecasterConfig(
        # Adapter width from the COLLATED batch (R12 one-hot prepended at
        # collation) — mirrors training/native_trainer.py.
        f_native=int(batch.hist.shape[2]),
        d_ctx=D_CTX,
        forecast_horizon=H,
        sequence_length=L,
        d_in=16,
        adapter_hidden_dim=16,
        type_embedding_dim=4,
        head_d_model=16,
        head_n_heads=2,
        head_num_layers=1,
        head_dropout=0.0,
    )
    return NativeEpiForecaster(config), batch


def test_raw_model_outputs_score_and_backprop_through_native_loss() -> None:
    model, batch = _build_model_and_batch()
    model.eval()
    outputs = model(batch)
    loss = NativeJointInferenceLoss(horizon_days=H, observation_loss="nll")
    components = loss(outputs, batch)

    # R1 on real outputs: cases distributions carry the nowcast width while
    # the routed daily targets are H-wide; the loss aligns internally.
    assert tuple(outputs["dist_cases_loc"].shape) == (3, H + 1)
    assert tuple(batch.head_target["cases"].shape) == (1, H)
    assert tuple(outputs["dist_deaths_loc"].shape) == (3, H)
    assert tuple(outputs["dist_hosp_weekly_loc"].shape) == (1, W_MAX)

    assert torch.isfinite(components["total"])
    for head in ("cases", "hosp", "deaths"):
        assert torch.isfinite(components[head]), head
        assert components[f"{head}_n_eff"].item() > 0, head
    assert components["cases_n_eff"].item() == float(H - 1)  # one masked slot
    assert components["deaths_n_eff"].item() == float(H)
    assert components["hosp_n_eff"].item() == 1.0  # second week partial

    components["total"].backward()
    for name, module in (
        ("input_adapter", model.input_adapter),
        ("backbone", model.base.backbone),
        ("cases_head", model.base.cases_head),
        ("deaths_head", model.base.deaths_head),
        ("hosp_head", model.base.hosp_head),
    ):
        grads = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert grads, f"no gradients reached {name}"
        for grad in grads:
            assert torch.isfinite(grad).all(), name
        assert any(float(grad.norm()) > 0 for grad in grads), name
