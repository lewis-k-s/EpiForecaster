"""Prepare exploratory neighbor timelines for the mobility report.

The selected examples are post hoc illustrations. This script does not change
the campaign scores or the immutable evaluation artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "outputs/loss_research/loss-type-full-coverage-v2"
DATASET = (
    ROOT / "outputs/loss_research/loss-type-research-v1/real_loss_fit20200930_v1.nc"
)
GRANULAR = CAMPAIGN / "source_resolution_input/r14_training"
REUSED = CAMPAIGN / "reuse_evidence_v1"
ABS_FORECASTS = (
    CAMPAIGN
    / "source_resolution_bundle/source_resolution_analysis"
    / "hospital_native_source_forecasts.csv"
)
ABS_SOURCE = (
    ROOT.parent / "dasymetric_mob/data/raw/COVID-19__Persones_hospitalitzades.csv"
)
ABS_MAPPING = ROOT.parent / "dasymetric_mob/data/output/abs_to_municipality_mapping.csv"
OUTPUT = ROOT / "outputs/reports/loss_type_full_coverage_v2_neighbor_stories"

STORIES = (
    {
        "story": "Barberà del Vallès cases",
        "target": "cases",
        "region_id": "08252",
        "neighbors": (("08187", "Sabadell"), ("08266", "Cerdanyola del Vallès")),
    },
    {
        "story": "Amposta wastewater",
        "target": "wastewater",
        "region_id": "43014",
        "neighbors": (("43155", "Tortosa"),),
    },
    {
        "story": "El Prat ABS 183 hospitalizations",
        "target": "hospitalizations",
        "region_id": "08169",
        "abs_code": "183",
        "neighbors": (("08019", "Barcelona"), ("08200", "Sant Boi de Llobregat")),
    },
)


def paired_summary(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["configuration"] = frame["condition"].str.replace(
        r":mobility_(?:on|off)$", "", regex=True
    )
    frame["mobility"] = frame["condition"].str.endswith(":mobility_on")
    index = ["story", "target_date", "fold", "seed", "configuration"]
    paired = frame.pivot(index=index, columns="mobility", values=score).dropna()
    paired.columns = ["off", "on"]
    paired["gain"] = paired["off"] - paired["on"]
    return (
        paired.groupby(level="story")
        .agg(
            paired_rows=("gain", "size"),
            mean_gain=("gain", "mean"),
            fraction_improved=("gain", lambda values: (values > 0).mean()),
        )
        .reset_index()
    )


def identity_shape_review(frame: pd.DataFrame, *, peak_week: int) -> pd.DataFrame:
    rows = []
    for (story, condition, seed), group in frame.groupby(
        ["story", "condition", "seed"]
    ):
        weekly = group.groupby("week").agg(
            observed=("observed", "mean"), forecast=("prediction", "mean")
        )
        if weekly.index.tolist() != [0, 1, 2, 3]:
            raise ValueError(f"Incomplete weekly trajectory for {story}: {condition}")
        rows.append(
            {
                "story": story,
                "condition": condition,
                "seed": seed,
                "mobility": condition.endswith(":mobility_on"),
                "observed_rise_to_peak": weekly.loc[peak_week, "observed"]
                - weekly.loc[0, "observed"],
                "forecast_rise_to_peak": weekly.loc[peak_week, "forecast"]
                - weekly.loc[0, "forecast"],
                "observed_change_after_peak": weekly.loc[3, "observed"]
                - weekly.loc[peak_week, "observed"],
                "forecast_change_after_peak": weekly.loc[3, "forecast"]
                - weekly.loc[peak_week, "forecast"],
            }
        )
    return pd.DataFrame(rows)


def municipal_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in GRANULAR.glob("*/*/test_granular.parquet"):
        identity = re.search(r"_f(\d+)_s(\d+)$", path.parent.name)
        if identity is None:
            raise ValueError(f"Cannot read fold and seed from {path}")
        frame = pd.read_parquet(
            path,
            columns=[
                "target",
                "region_id",
                "target_date",
                "observed",
                "prediction",
                "wis",
            ],
        )
        frame["condition"] = path.parent.parent.name.split("__")[-1]
        frame["fold"] = int(identity.group(1))
        frame["seed"] = int(identity.group(2))
        frames.append(frame)
    for path in REUSED.glob("*/artifacts/evaluation/calibration/test_granular.parquet"):
        identity = re.fullmatch(
            r"(normal|student_t)-(crps|nll)-m([01])-f(\d+)-s(\d+)-r\d+",
            path.parents[3].name,
        )
        if identity is None:
            raise ValueError(f"Cannot read reused identity from {path}")
        frame = pd.read_parquet(
            path,
            columns=[
                "target",
                "region_id",
                "target_date",
                "observed",
                "prediction",
                "wis",
            ],
        )
        mobility = "on" if identity.group(3) == "1" else "off"
        frame["condition"] = (
            f"{identity.group(1)}:{identity.group(2)}:mobility_{mobility}"
        )
        frame["fold"] = int(identity.group(4))
        frame["seed"] = int(identity.group(5))
        frames.append(frame)
    if len(frames) != 160:
        raise ValueError(f"Expected 160 forecast identities, found {len(frames)}")
    all_rows = pd.concat(frames, ignore_index=True)
    selected = []
    for story in STORIES[:2]:
        rows = all_rows.loc[
            all_rows["target"].eq(story["target"])
            & all_rows["region_id"].eq(story["region_id"])
        ].copy()
        rows["story"] = story["story"]
        selected.append(rows)
    selected = pd.concat(selected, ignore_index=True)
    score_summary = paired_summary(selected, "wis")
    expected_rows = {
        "Barberà del Vallès cases": 224,
        "Amposta wastewater": 32,
    }
    for row in score_summary.itertuples(index=False):
        if row.paired_rows != expected_rows[row.story]:
            raise ValueError(f"Incomplete paired score support for {row.story}")
    selected["mobility"] = selected["condition"].str.endswith(":mobility_on")
    selected["target_date"] = pd.to_datetime(selected["target_date"])
    first_date = selected.groupby("story")["target_date"].transform("min")
    selected["week"] = ((selected["target_date"] - first_date).dt.days // 7).astype(int)
    timeline = selected.groupby(["story", "week", "mobility"], as_index=False).agg(
        observed=("observed", "mean"),
        forecast=("prediction", "mean"),
        first_date=("target_date", "min"),
        last_date=("target_date", "max"),
        rows=("prediction", "size"),
    )
    timeline["unit"] = timeline["story"].map(
        {
            "Barberà del Vallès cases": "log1p cases per 100,000",
            "Amposta wastewater": "mean log1p biomarker signal",
        }
    )
    shape = pd.concat(
        [
            identity_shape_review(
                selected.loc[selected["story"].eq(story["story"])],
                peak_week=1 if story["target"] == "cases" else 2,
            )
            for story in STORIES[:2]
        ],
        ignore_index=True,
    )
    return timeline, score_summary, shape


def hospital_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    story = STORIES[2]
    frame = pd.read_csv(ABS_FORECASTS, dtype={"abs_code": str})
    frame = frame.loc[frame["abs_code"].eq(story["abs_code"])].copy()
    if frame.empty or frame["fold"].nunique() != 1:
        raise ValueError("ABS 183 forecast support is missing or spans multiple folds")
    frame["story"] = story["story"]
    score_summary = paired_summary(frame, "rate_abs_error")
    if score_summary["paired_rows"].tolist() != [32]:
        raise ValueError("Incomplete paired ABS 183 forecast support")
    frame["mobility"] = frame["condition"].str.endswith(":mobility_on")
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["week"] = (
        (frame["target_date"] - frame["target_date"].min()).dt.days // 7
    ).astype(int)
    timeline = (
        frame.groupby(["story", "week", "mobility"], as_index=False)
        .agg(
            observed=("observed_count", "mean"),
            forecast=("prediction_count", "mean"),
            first_date=("target_date", "min"),
            last_date=("target_date", "max"),
            rows=("prediction_count", "size"),
        )
        .assign(unit="ABS weekly hospital counts")
    )
    shape_frame = frame.rename(
        columns={"observed_count": "observed", "prediction_count": "prediction"}
    )
    shape = identity_shape_review(shape_frame, peak_week=2)
    return timeline, score_summary, shape


def neighbor_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = xr.open_dataset(DATASET)
    dates = pd.to_datetime(dataset.date.values)
    ids = dataset.region_id.values.astype(str)
    code_to_index = {code: index for index, code in enumerate(ids)}
    selected_codes = list(
        dict.fromkeys(
            code
            for story in STORIES
            for code in (story["region_id"], *(code for code, _ in story["neighbors"]))
        )
    )
    for code in selected_codes:
        if code not in code_to_index:
            raise ValueError(f"Region {code} is absent from the campaign dataset")
    case_values = dataset["cases"].sel(run_id="real", region_id=selected_codes).values
    case_masks = (
        dataset["cases_mask"].sel(run_id="real", region_id=selected_codes).values
    )
    cases = pd.DataFrame(case_values, index=dates, columns=selected_codes).where(
        pd.DataFrame(case_masks, index=dates, columns=selected_codes).eq(1)
    )
    cases = cases.loc["2020-11-01":"2020-12-30"]
    mobility = dataset["mobility"].sel(
        run_id="real",
        date=slice("2020-12-01", "2020-12-29"),
        destination=[story["region_id"] for story in STORIES],
    )
    mean_raw_flow = np.expm1(mobility.values).mean(axis=0)
    history = []
    links = []
    for story_index, story in enumerate(STORIES):
        target_code = story["region_id"]
        sources = ((target_code, "Target municipality"), *story["neighbors"])
        for source_code, name in sources:
            series = cases[source_code]
            trailing_mean = series.rolling(7, min_periods=4).mean()
            for date, value in series.items():
                history.append(
                    {
                        "story": story["story"],
                        "region_id": source_code,
                        "name": name,
                        "role": "target"
                        if source_code == target_code
                        else "incoming neighbor",
                        "date": date,
                        "cases_log1p_rate_observed": value,
                        "cases_7day_mean": trailing_mean.loc[date],
                        "observed": pd.notna(value),
                    }
                )
            if source_code == target_code:
                continue
            flow = float(mean_raw_flow[code_to_index[source_code], story_index])
            if flow < 10:
                raise ValueError(
                    f"Selected mobility link is below the model threshold: {source_code}"
                )
            links.append(
                {
                    "story": story["story"],
                    "origin_id": source_code,
                    "origin_name": name,
                    "destination_id": target_code,
                    "mean_incoming_trips_per_day": flow,
                    "period": "2020-12-01/2020-12-29",
                }
            )
    dataset.close()
    return pd.DataFrame(history), pd.DataFrame(links)


def target_history_and_anchors(
    neighbor_cases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the preforecast target scale visible beside each held-out forecast."""
    dataset = xr.open_dataset(DATASET).sel(run_id="real")
    history = []
    anchors = []
    case_story = STORIES[0]
    case_rows = neighbor_cases.loc[
        neighbor_cases["story"].eq(case_story["story"])
        & neighbor_cases["role"].eq("target")
    ]
    for row in case_rows.itertuples(index=False):
        history.append(
            {
                "story": case_story["story"],
                "date": row.date,
                "observed": row.cases_log1p_rate_observed,
                "seven_day_mean": row.cases_7day_mean,
                "unit": "log1p cases per 100,000",
            }
        )
    case_values = dataset["cases"].sel(region_id=case_story["region_id"]).to_series()
    case_mask = dataset["cases_mask"].sel(region_id=case_story["region_id"]).to_series()
    case_values = case_values.where(case_mask.eq(1)).loc[:"2020-12-30"].dropna()
    anchors.append(
        {
            "story": case_story["story"],
            "date": case_values.index[-1],
            "value": case_values.iloc[-1],
            "basis": "last observed municipal target",
        }
    )

    ww_story = STORIES[1]
    ww_components = []
    for name in ("edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4"):
        values = dataset[name].sel(region_id=ww_story["region_id"]).to_series()
        mask = dataset[f"{name}_mask"].sel(region_id=ww_story["region_id"]).to_series()
        ww_components.append(values.where(mask.eq(1)))
    ww_values = (
        pd.concat(ww_components, axis=1).mean(axis=1).loc[:"2020-12-30"].dropna()
    )
    for date, value in ww_values.loc["2020-11-01":"2020-12-30"].items():
        history.append(
            {
                "story": ww_story["story"],
                "date": date,
                "observed": value,
                "seven_day_mean": np.nan,
                "unit": "mean log1p biomarker signal",
            }
        )
    anchors.append(
        {
            "story": ww_story["story"],
            "date": ww_values.index[-1],
            "value": ww_values.iloc[-1],
            "basis": "last observed municipal target",
        }
    )

    hospital_story = STORIES[2]
    source = pd.read_csv(
        ABS_SOURCE,
        usecols=["data_inici", "codi_abs", "casos"],
        dtype={"codi_abs": str},
    )
    source = source.loc[
        source["codi_abs"].str.zfill(3).eq(hospital_story["abs_code"])
    ].copy()
    source["date"] = pd.to_datetime(
        source["data_inici"], format="%d/%m/%Y"
    ) + pd.Timedelta(days=7)
    source = source.groupby("date")["casos"].sum().loc["2020-11-01":"2020-12-30"]
    for date, value in source.items():
        history.append(
            {
                "story": hospital_story["story"],
                "date": date,
                "observed": value,
                "seven_day_mean": np.nan,
                "unit": "native ABS weekly hospital counts",
            }
        )
    mapping = pd.read_csv(
        ABS_MAPPING, dtype={"abs_code": str, "municipality_code": str}
    )
    mapping = mapping.loc[
        mapping["abs_code"].str.zfill(3).eq(hospital_story["abs_code"])
    ]
    last_date = pd.Timestamp("2020-12-28")
    projected_count = 0.0
    for row in mapping.itertuples(index=False):
        code = row.municipality_code
        value = float(
            dataset["hospitalizations"].sel(date=last_date, region_id=code).item()
        )
        observed = int(
            dataset["hospitalizations_mask"].sel(date=last_date, region_id=code).item()
        )
        if observed != 1:
            raise ValueError(f"Missing projected hospital input for {code}")
        population = float(dataset["population"].sel(region_id=code).item())
        projected_count += np.expm1(value) * population / 1e5 * row.weight
    anchors.append(
        {
            "story": hospital_story["story"],
            "date": last_date,
            "value": projected_count,
            "basis": "processed municipal input re-projected to ABS",
        }
    )
    dataset.close()
    return pd.DataFrame(history), pd.DataFrame(anchors)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    municipal, municipal_scores, municipal_shape = municipal_forecasts()
    hospital, hospital_scores, hospital_shape = hospital_forecasts()
    history, links = neighbor_history()
    target_history, anchors = target_history_and_anchors(history)
    pd.concat([municipal, hospital], ignore_index=True).to_csv(
        OUTPUT / "forecast_timeline.csv", index=False
    )
    pd.concat([municipal_scores, hospital_scores], ignore_index=True).to_csv(
        OUTPUT / "paired_score_summary.csv", index=False
    )
    pd.concat([municipal_shape, hospital_shape], ignore_index=True).to_csv(
        OUTPUT / "identity_shape_review.csv", index=False
    )
    history.to_csv(OUTPUT / "neighbor_case_history.csv", index=False)
    target_history.to_csv(OUTPUT / "target_history.csv", index=False)
    anchors.to_csv(OUTPUT / "target_anchors.csv", index=False)
    links.to_csv(OUTPUT / "incoming_links.csv", index=False)
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            {
                "selection": "Post hoc examples selected for visible neighbor signals and paired forecast gains",
                "forecast_origin": "2020-12-30",
                "input_history": "2020-11-01/2020-12-30",
                "link_flow_summary": "2020-12-01/2020-12-29",
                "campaign_forecasts": str(GRANULAR.relative_to(ROOT)),
                "abs_reprojection": str(ABS_FORECASTS.relative_to(ROOT)),
                "campaign_dataset": str(DATASET.relative_to(ROOT)),
                "hospital_abs_source": str(ABS_SOURCE),
                "hospital_abs_mapping": str(ABS_MAPPING),
                "score_metrics": {"municipal": "WIS", "hospital_abs": "log1p rate MAE"},
            },
            indent=2,
        )
        + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
