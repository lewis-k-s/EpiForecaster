import pandas as pd

from dataviz.ablation_plots import _prepare_model_order


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
