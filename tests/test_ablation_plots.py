import pandas as pd

from pathlib import Path

import pytest

from dataviz.ablation_plots import (
    _prepare_model_order,
    plot_ablation_delta_histograms,
    plot_seed_matched_delta_diagnostics,
)


def test_prepare_model_order_clusters_ablations() -> None:
    df = pd.DataFrame(
        {
            "model": [
                "sig:hosp:proxy",
                "regions:off",
                "baseline",
                "kernel:fixed",
                "sig:ww:aux",
                "residual:off",
                "mobility:off",
                "sig:cases:off",
                "context:off",
                "sir:off",
                "gradnorm:on",
                "sig:deaths:aux",
            ],
            "target": ["cases"] * 12,
        }
    )

    assert _prepare_model_order(df, baseline_name="baseline") == [
        "baseline",
        "mobility:off",
        "regions:off",
        "context:off",
        "residual:off",
        "sir:off",
        "sig:ww:aux",
        "sig:cases:off",
        "sig:hosp:proxy",
        "sig:deaths:aux",
        "gradnorm:on",
        "kernel:fixed",
    ]


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib not installed",
)
def test_plot_ablation_delta_histograms_writes_metric_plot(tmp_path: Path) -> None:
    pairwise_csv = tmp_path / "ablation_metrics_deltas_seed_pairwise.csv"
    pd.DataFrame(
        [
            {
                "model": "sig:cases:aux",
                "target": "cases",
                "seed": 1,
                "mae_delta_pct": 10.0,
            },
            {
                "model": "sig:cases:aux",
                "target": "cases",
                "seed": 2,
                "mae_delta_pct": -5.0,
            },
            {
                "model": "sig:ww:proxy",
                "target": "wastewater",
                "seed": 1,
                "mae_delta_pct": 20.0,
            },
        ]
    ).to_csv(pairwise_csv, index=False)

    output_path = plot_ablation_delta_histograms(pairwise_csv, output_dir=tmp_path)

    assert output_path == tmp_path / "ablation_delta_histogram_mae.png"
    assert output_path.exists()


@pytest.mark.skipif(
    not pytest.importorskip("matplotlib", reason="matplotlib not installed"),
    reason="matplotlib not installed",
)
def test_plot_seed_matched_delta_diagnostics_writes_heatmaps_and_histograms(
    tmp_path: Path,
) -> None:
    summary_csv = tmp_path / "ablation_metrics_deltas_seed_matched.csv"
    pairwise_csv = tmp_path / "ablation_metrics_deltas_seed_pairwise.csv"
    pd.DataFrame(
        [
            {
                "model": "sig:cases:aux",
                "target": "cases",
                "mae_delta_pct_mean": 10.0,
                "rmse_delta_pct_mean": 12.0,
            },
            {
                "model": "sig:ww:proxy",
                "target": "wastewater",
                "mae_delta_pct_mean": -3.0,
                "rmse_delta_pct_mean": -4.0,
            },
        ]
    ).to_csv(summary_csv, index=False)
    pd.DataFrame(
        [
            {
                "model": "sig:cases:aux",
                "target": "cases",
                "seed": 1,
                "mae_delta_pct": 9.0,
                "rmse_delta_pct": 11.0,
            },
            {
                "model": "sig:ww:proxy",
                "target": "wastewater",
                "seed": 1,
                "mae_delta_pct": -2.0,
                "rmse_delta_pct": -5.0,
            },
        ]
    ).to_csv(pairwise_csv, index=False)

    artifacts = plot_seed_matched_delta_diagnostics(
        summary_csv,
        pairwise_csv,
        output_dir=tmp_path,
        metrics=["mae", "rmse"],
    )

    assert set(artifacts) == {
        "heatmap_mae",
        "histogram_mae",
        "heatmap_rmse",
        "histogram_rmse",
    }
    for path in artifacts.values():
        assert path.exists()
