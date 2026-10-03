"""Single-run figures from canonical, saved report evidence only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HEADS = {
    "cases": ("Cases", "cases", "Cases / day"),
    "hospitalizations": ("Admissions", "hosp", "Admissions / week"),
    "deaths": ("All-cause deaths", "deaths", "Deaths / day"),
    "wastewater": ("Wastewater", "ww", "Source concentration units"),
}
COLORS = {"train": "#2563a6", "val": "#d87919", "test": "#7c4aa5"}


class RunFigures:
    def __init__(self, output: Path):
        self.output = output
        self.assets: dict[str, str] = {}
        (output / "figures").mkdir(parents=True, exist_ok=True)

    def save(self, key: str, figure) -> None:
        figure.tight_layout()
        figure.savefig(
            self.output / "figures" / f"{key}.png", dpi=130, facecolor="white"
        )
        plt.close(figure)
        self.assets[key] = f"figures/{key}.png"

    @staticmethod
    def epoch_axis(ax, train: pd.DataFrame, selected_epoch: int | None) -> None:
        if "batch_size" in train:
            changes = train.batch_size.ne(train.batch_size.shift())
            for row in train[changes].iloc[1:].itertuples(index=False):
                ax.axvline(row.epoch - 0.5, linestyle=":", color="#8d6e35")
                ax.text(
                    row.epoch - 0.5,
                    0.98,
                    f"Batch {row.batch_size}",
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    fontsize=8,
                )
        if selected_epoch is not None:
            ax.axvline(selected_epoch, linestyle=":", color="#555555", alpha=0.5)
        ax.set_xlabel("Completed epoch")
        ax.grid(alpha=0.2)

    def epochs(
        self,
        frames: dict[str, pd.DataFrame],
        selected_epoch: int | None,
        checkpoint_wis: float | None,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
        train = frames["train"]
        if "train_objective" in train:
            axes[0].plot(
                train.epoch,
                train.train_objective,
                "o-",
                color=COLORS["train"],
                label="train",
            )
        for split in ("val", "test"):
            frame = frames[split]
            if f"{split}_wis" in frame:
                axes[1].plot(
                    frame.epoch,
                    frame[f"{split}_wis"],
                    "o-",
                    color=COLORS[split],
                    label=split,
                )
        if selected_epoch is not None and checkpoint_wis is not None:
            axes[1].scatter(
                [selected_epoch],
                [checkpoint_wis],
                marker="D",
                color=COLORS["test"],
                label="test · selected checkpoint",
            )
        axes[0].set_title("train_objective · recorded epochs")
        axes[1].set_title("val_wis / test_wis · equal-head mean")
        for ax in axes:
            self.epoch_axis(ax, train, selected_epoch)
            if ax.lines or ax.collections:
                ax.legend(fontsize=8)
        self.save("loss_epochs", fig)

    def wastewater(
        self, frames: dict[str, pd.DataFrame], selected_epoch: int | None
    ) -> None:
        fields = ("ww_mixed_objective", "censored_nll_ww")
        if not any(field in frame for frame in frames.values() for field in fields):
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
        for ax, field, label in zip(
            axes,
            fields,
            ("Mixed wastewater objective", "Censored wastewater NLL"),
            strict=True,
        ):
            for split, frame in frames.items():
                if field in frame:
                    ax.plot(
                        frame.epoch,
                        frame[field],
                        "o-",
                        color=COLORS[split],
                        label=split,
                    )
            ax.set_title(label)
            self.epoch_axis(ax, frames["train"], selected_epoch)
            if ax.lines:
                ax.legend(fontsize=8)
        self.save("wastewater_objectives", fig)

    def heads(
        self,
        frames: dict[str, pd.DataFrame],
        scores: pd.DataFrame,
        selected_epoch: int | None,
    ) -> None:
        for metric in ("crps", "wis", "nll"):
            if not any(
                f"{metric}_{head}" in f
                for f in frames.values()
                for _, head, _ in HEADS.values()
            ):
                continue
            fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
            for ax, (target, (label, head, _)) in zip(
                axes.flat, HEADS.items(), strict=True
            ):
                for split, frame in frames.items():
                    if f"{metric}_{head}" in frame:
                        ax.plot(
                            frame.epoch,
                            frame[f"{metric}_{head}"],
                            "o-",
                            color=COLORS[split],
                            label=split,
                        )
                if selected_epoch is not None and not scores.empty and metric in scores:
                    test = scores[(scores.split == "test") & (scores.target == target)]
                    if not test.empty:
                        ax.scatter(
                            [selected_epoch],
                            test[metric],
                            marker="D",
                            color=COLORS["test"],
                            label="test · selected checkpoint",
                        )
                ax.set_title(label)
                ax.set_ylabel(metric.upper() + " · target score space")
                self.epoch_axis(ax, frames["train"], selected_epoch)
                if ax.lines or ax.collections:
                    ax.legend(fontsize=8)
            self.save(f"heads_{metric}_epochs", fig)

    def horizons(self, scores: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
        for column, (target, (label, _, _)) in enumerate(HEADS.items()):
            for split in ("val", "test"):
                frame = scores[
                    (scores.split == split) & (scores.target == target)
                ].sort_values("horizon")
                axes[0, column].plot(
                    frame.horizon,
                    frame.wis,
                    "o-",
                    color=COLORS[split],
                    label=split,
                    markersize=2,
                )
                axes[1, column].plot(
                    frame.horizon,
                    frame.coverage_80,
                    "o-",
                    color=COLORS[split],
                    label=split,
                    markersize=2,
                )
            axes[0, column].set_title(label)
            axes[0, column].set_ylabel("WIS · score space")
            axes[1, column].set_ylabel("80% interval coverage")
            axes[1, column].axhline(0.8, linestyle=":", color="#555555")
            axes[1, column].set_ylim(0, 1.05)
            for ax in axes[:, column]:
                ax.set_xlabel(
                    "Horizon (weeks)"
                    if target == "hospitalizations"
                    else "Horizon (days)"
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        self.save("horizon_scores", fig)

    def dynamics(self, samples: pd.DataFrame) -> None:
        fields = [
            ("train_objective", "train_loss", "loss_train_step"),
            ("grad_norm", "gradnorm_total_preclip"),
        ]
        available = [
            next((field for field in group if field in samples), None)
            for group in fields
        ]
        if not any(available):
            return
        fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
        for ax, field, label in zip(
            axes,
            available,
            ("Sampled train_objective", "Pre-clip gradient norm"),
            strict=True,
        ):
            if field:
                ax.plot(samples.step, samples[field], linewidth=0.7, color="#2563a6")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Saved global optimizer step")
        self.save("sampled_loss_gradients", fig)

    def gpu(self, key: str, samples: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 6))
        for ax, field, label in zip(
            axes.flat,
            (
                "gpu_utilization_percent",
                "memory_used_mib",
                "power_watts",
                "temperature_celsius",
            ),
            ("GPU activity (%)", "Memory (MiB)", "Power (W)", "Temperature (°C)"),
            strict=True,
        ):
            for uuid, group in samples.groupby("gpu_uuid", sort=True):
                ax.plot(
                    pd.to_datetime(group.timestamp_utc, utc=True),
                    pd.to_numeric(group[field], errors="coerce"),
                    label=uuid,
                    linewidth=0.8,
                )
            ax.set_ylabel(label)
            ax.set_xlabel("UTC · full worker invocation")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=7)
        axes[0, 0].set_ylim(0, 100)
        self.save(key, fig)

    def examples(
        self,
        split: str,
        rows: pd.DataFrame,
        choices: pd.DataFrame,
        history: pd.DataFrame,
        excess: pd.DataFrame,
        context: dict,
    ) -> None:
        for target, (label, _, units) in HEADS.items():
            selected = choices[choices.target == target]
            if selected.empty:
                continue
            fig, axes = plt.subplots(
                2, len(selected), figsize=(5 * len(selected), 7), squeeze=False
            )
            for column, choice in enumerate(selected.itertuples(index=False)):
                example = rows[
                    (rows.target == target)
                    & (rows.source_id == choice.source_id)
                    & (rows.window_start == choice.window_start)
                ].sort_values("target_date")
                origin = pd.Timestamp(choice.origin_date)
                start = origin - pd.Timedelta(days=context["input_window_length"] - 1)
                past = history[
                    (history.target == target)
                    & (history.source_id == choice.source_id)
                    & (pd.to_datetime(history.input_date) >= start)
                    & (pd.to_datetime(history.input_date) <= origin)
                ]
                dates = pd.to_datetime(example.target_date)
                for ax, hist, obs, pred in (
                    (axes[0, column], "observed_raw", "observed_raw", "prediction_raw"),
                    (axes[1, column], "observed_score", "observed", "prediction"),
                ):
                    ax.plot(
                        pd.to_datetime(past.date),
                        past[hist].where(~past.censored),
                        "o-",
                        color="#747b83",
                        markersize=2,
                        label="Observed history",
                    )
                    bounds = past[past.censored]
                    if not bounds.empty:
                        ax.scatter(
                            pd.to_datetime(bounds.date),
                            bounds[hist],
                            marker="v",
                            facecolors="none",
                            edgecolors="#747b83",
                            label="Censored bound",
                        )
                    ax.plot(
                        dates,
                        example[obs],
                        "o-",
                        color=COLORS["train"],
                        markersize=3,
                        label="Observed future",
                    )
                    ax.plot(
                        dates,
                        example[pred],
                        "o--",
                        color=COLORS["val"],
                        markersize=2,
                        label="Forecast median",
                    )
                    ax.axvline(
                        origin + pd.Timedelta(hours=12), color="#333333", linewidth=1
                    )
                    ax.set_xlim(
                        start,
                        origin + pd.Timedelta(days=context["forecast_horizon"] + 1),
                    )
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(alpha=0.2)
                axes[0, column].set_title(
                    f"{choice.example_rank} · {choice.source_id}\nOrigin {origin:%Y-%m-%d}"
                )
                axes[0, column].set_ylabel(units)
                axes[1, column].set_ylabel("Target score space")
                for level, alpha in ((95, 0.12), (80, 0.25)):
                    if f"interval_low_{level}" in example:
                        axes[1, column].fill_between(
                            dates,
                            example[f"interval_low_{level}"],
                            example[f"interval_high_{level}"],
                            color=COLORS["val"],
                            alpha=alpha,
                            label=f"{level}% interval",
                        )
            axes[0, 0].legend(fontsize=8)
            axes[1, 0].legend(fontsize=8)
            fig.suptitle(
                f"{split} · {label} · {context['input_window_length']}-day history / {context['forecast_horizon']}-day horizon"
            )
            self.save(f"examples_{split}_{target}", fig)
        deaths = choices[choices.target == "deaths"]
        if deaths.empty or excess.empty:
            return
        fig, axes = plt.subplots(
            1, len(deaths), figsize=(5 * len(deaths), 3.7), squeeze=False
        )
        for ax, choice in zip(axes[0], deaths.itertuples(index=False), strict=True):
            example = excess[
                (excess.abs_id == choice.source_id)
                & (excess.origin_index == choice.origin_index)
            ].sort_values("target_date")
            origin = pd.Timestamp(choice.origin_date)
            start = origin - pd.Timedelta(days=context["input_window_length"] - 1)
            past = history[
                (history.target == "deaths")
                & (history.source_id == choice.source_id)
                & (history.input_date >= start)
                & (history.input_date <= origin)
            ]
            ax.plot(
                past.date,
                past.excess,
                "o-",
                color="#747b83",
                markersize=2,
                label="Observed history",
            )
            ax.plot(
                pd.to_datetime(example.target_date),
                example.observed_excess,
                "o-",
                label="Observed excess",
            )
            ax.plot(
                pd.to_datetime(example.target_date),
                example.predicted_excess,
                "o--",
                label="Predicted excess",
            )
            ax.axhline(0, linewidth=0.7, color="#333333")
            ax.axvline(origin + pd.Timedelta(hours=12), color="#333333", linewidth=1)
            ax.set_title(f"{choice.example_rank} · ABS {choice.source_id}")
            ax.set_ylabel("Signed excess deaths / day")
            ax.tick_params(axis="x", rotation=30)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[0, 0].legend(fontsize=8)
        self.save(f"examples_{split}_excess_deaths", fig)
