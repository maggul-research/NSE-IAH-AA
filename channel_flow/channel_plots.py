#!/usr/bin/env python3
"""Create the channel-flow performance and residual plots used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ALL_ALGORITHMS = [
    "IAH",
    "AA-IAH(m=1)",
    "AA-IAH(m=2)",
    "AA-IAH(m=3)",
    "AA-IAH(m=4)",
    "AA-IAH(m=5)",
]
PAPER_RESIDUAL_ALGORITHMS = ["IAH", "AA-IAH(m=5)"]

TITLE_FS = 46
LABEL_FS = 46
TICK_FS = 38
TEXT_FS = 36
XTICK_ALGO_FS = 42
RESID_TICK_FS = 42
RESID_LEGEND_FS = 40

BAR_FIG_W, BAR_FIG_H = 10, 6
RESID_FIG_W, RESID_FIG_H = 10, 6
PANEL_LEFT, PANEL_RIGHT = 0.20, 0.96
BAR_MARGINS = dict(left=PANEL_LEFT, right=PANEL_RIGHT, top=0.88, bottom=0.10)
RESID_MARGINS = dict(left=PANEL_LEFT, right=PANEL_RIGHT, top=0.90, bottom=0.22)

ITER_YLIM_FACTOR = 1.55
CPU_YLIM_FACTOR = 1.90
RESID_YMIN = 5e-7
RESID_LABELED_EXPONENTS = [0, -3, -6]

COLOR_ITER = "#3F37C9"
COLOR_CPU = "#FFB703"
COLOR_IAH_LINE = "#B22222"
COLOR_AA_LINE = "#000080"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read every rho_* result directory produced by run_channel.py and "
            "write the six PDF panels used by the channel-flow figure."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory containing rho_*/ CSV results (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../../plots"),
        help="plot output directory (default: plots beside the build directory)",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=100.0,
        help="Reynolds number to plot (default: 100)",
    )
    return parser.parse_args()


def format_time(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining = total_seconds % 60
    if hours > 0:
        return f"{hours}h:{minutes}m:{remaining}s"
    if minutes > 0:
        return f"{minutes}m:{remaining}s"
    return f"{remaining}s"


def simplify_algorithm_label(label: str) -> str:
    label = str(label)
    if label.startswith("AA-IAH(") and label.endswith(")"):
        return label[len("AA-IAH(") : -1]
    return label


def sci_notation_formatter(value: float, _position: float) -> str:
    if value <= 0:
        return ""
    exponent = int(np.round(np.log10(value)))
    if exponent not in RESID_LABELED_EXPONENTS:
        return ""
    return f"1e{exponent}"


def number_label(value: float) -> str:
    return f"{value:g}"


def rho_from_directory(directory: Path) -> float:
    token = directory.name.removeprefix("rho_")
    token = token.replace("minus_", "-").replace("p", ".")
    return float(token)


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []

    performance_paths = sorted(results_dir.glob("rho_*/performance_summary.csv"))
    if not performance_paths:
        raise FileNotFoundError(
            f"No rho_*/performance_summary.csv files found under {results_dir}"
        )

    for performance_path in performance_paths:
        residual_path = performance_path.with_name("residue_history.csv")
        if not residual_path.is_file():
            raise FileNotFoundError(f"Missing matching residual file: {residual_path}")

        performance = pd.read_csv(performance_path)
        residual = pd.read_csv(residual_path)


        performance_frames.append(performance)
        residual_frames.append(residual)

    performance = pd.concat(performance_frames, ignore_index=True)
    residual = pd.concat(residual_frames, ignore_index=True)

    required_performance = {
        "Re",
        "rho",
        "m",
        "Algorithm",
        "TotalIterations",
        "CPUTime",
    }
    required_residual = {
        "Re",
        "rho",
        "m",
        "Algorithm",
        "Iteration",
        "RelativeError",
    }
    missing_performance = required_performance.difference(performance.columns)
    missing_residual = required_residual.difference(residual.columns)
    if missing_performance:
        raise ValueError(
            "Performance CSV is missing columns: "
            + ", ".join(sorted(missing_performance))
        )
    if missing_residual:
        raise ValueError(
            "Residual CSV is missing columns: " + ", ".join(sorted(missing_residual))
        )

    return performance, residual


def select_reynolds_rows(data: pd.DataFrame, reynolds_number: float) -> pd.DataFrame:
    return data[np.isclose(data["Re"].astype(float), reynolds_number)].copy()


def verify_paper_algorithms(performance: pd.DataFrame, rho: float) -> None:
    counts = performance["Algorithm"].astype(str).value_counts()
    available = set(counts.index)
    missing = [algorithm for algorithm in ALL_ALGORITHMS if algorithm not in available]
    if missing:
        raise ValueError(
            f"rho={number_label(rho)} is missing paper runs: {', '.join(missing)}"
        )
    repeated = [algorithm for algorithm in ALL_ALGORITHMS if counts[algorithm] != 1]
    if repeated:
        raise ValueError(
            f"rho={number_label(rho)} has repeated runs: {', '.join(repeated)}; "
            "rerun with --overwrite"
        )


def draw_performance_plot(
    performance: pd.DataFrame,
    rho: float,
    output_dir: Path,
) -> Path:
    data = performance.copy()
    data["Algorithm"] = pd.Categorical(
        data["Algorithm"], categories=ALL_ALGORITHMS, ordered=True
    )
    data = data.sort_values("Algorithm")

    figure, left_axis = plt.subplots(figsize=(BAR_FIG_W, BAR_FIG_H))
    left_axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    left_axis.grid(False, axis="x")

    x_positions = np.arange(len(data))
    width = 0.4
    iteration_bars = left_axis.bar(
        x_positions - width / 2,
        data["TotalIterations"],
        width,
        label="Total Iterations",
        color=COLOR_ITER,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )
    right_axis = left_axis.twinx()
    right_axis.grid(False)
    time_bars = right_axis.bar(
        x_positions + width / 2,
        data["CPUTime"],
        width,
        label="CPU Time",
        color=COLOR_CPU,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )

    left_axis.set_xlabel("")
    left_axis.set_ylabel("")
    right_axis.set_ylabel("")
    left_axis.set_title(fr"$\rho = {number_label(rho)}$", fontsize=TITLE_FS)
    left_axis.set_xticks(x_positions)
    left_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    left_axis.tick_params(axis="y", left=False, labelleft=False)
    right_axis.tick_params(axis="y", right=False, labelright=False)

    left_axis.set_ylim(0, data["TotalIterations"].max() * ITER_YLIM_FACTOR)
    right_axis.set_ylim(0, data["CPUTime"].max() * CPU_YLIM_FACTOR)

    for bar in iteration_bars:
        value = bar.get_height()
        if not pd.isna(value):
            left_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + left_axis.get_ylim()[1] * 0.015,
                int(value),
                ha="center",
                va="bottom",
                fontsize=TEXT_FS,
                fontweight="normal",
                rotation=90,
                clip_on=True,
            )

    for bar in time_bars:
        value = bar.get_height()
        if not pd.isna(value):
            right_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + right_axis.get_ylim()[1] * 0.015,
                format_time(value),
                ha="center",
                va="bottom",
                fontsize=TEXT_FS,
                fontweight="normal",
                rotation=90,
                clip_on=True,
            )

    for axis in (left_axis, right_axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("lightgray")
            spine.set_linewidth(1.5)

    figure.subplots_adjust(**BAR_MARGINS)
    reynolds_number = number_label(float(data["Re"].iloc[0]))
    output_path = output_dir / (
        f"channel_performance_re_{reynolds_number}_rho_{number_label(rho)}.pdf"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def draw_residual_plot(
    residual: pd.DataFrame,
    rho: float,
    output_dir: Path,
    show_x_label: bool,
    show_y_label: bool,
) -> Path:
    figure, axis = plt.subplots(figsize=(RESID_FIG_W, RESID_FIG_H))
    axis.grid(True, which="both", linestyle="--", alpha=0.3)
    colors = {"IAH": COLOR_IAH_LINE, "AA-IAH(m=5)": COLOR_AA_LINE}

    for algorithm in PAPER_RESIDUAL_ALGORITHMS:
        data = residual[residual["Algorithm"] == algorithm].sort_values("Iteration")
        if data.empty:
            raise ValueError(
                f"rho={number_label(rho)} has no residual history for {algorithm}"
            )
        axis.plot(
            data["Iteration"],
            data["RelativeError"],
            label="IAH" if algorithm == "IAH" else "AA-IAH",
            color=colors[algorithm],
            linewidth=3.5,
            zorder=5 if algorithm == "IAH" else 3,
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation_formatter))
    axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=20))
    axis.set_ylim(bottom=RESID_YMIN)
    axis.set_xlabel("Iteration Count" if show_x_label else "", fontsize=LABEL_FS)
    axis.set_ylabel("", fontsize=LABEL_FS
    )
    axis.set_title(fr"$\rho = {number_label(rho)}$", fontsize=TITLE_FS)
    axis.tick_params(axis="both", labelsize=RESID_TICK_FS)
    if not show_y_label:
        axis.tick_params(axis="y", labelleft=False)
    axis.legend(
        fontsize=RESID_LEGEND_FS,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("lightgray")
        spine.set_linewidth(1.5)

    figure.subplots_adjust(**RESID_MARGINS)
    reynolds_number = number_label(float(residual["Re"].iloc[0]))
    output_path = output_dir / (
        f"channel_residual_history_re_{reynolds_number}_rho_{number_label(rho)}.pdf"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        performance, residual = load_results(results_dir)
        performance = select_reynolds_rows(performance, args.re)
        residual = select_reynolds_rows(residual, args.re)
        if performance.empty or residual.empty:
            raise ValueError(f"No results found for Re={number_label(args.re)}")

        rho_values = sorted(performance["rho"].astype(float).unique())
        output_dir.mkdir(parents=True, exist_ok=True)
        middle_rho = rho_values[len(rho_values) // 2]
        first_rho = rho_values[0]

        written: list[Path] = []
        for rho in rho_values:
            performance_for_rho = performance[
                np.isclose(performance["rho"].astype(float), rho)
            ].copy()
            residual_for_rho = residual[
                np.isclose(residual["rho"].astype(float), rho)
            ].copy()
            verify_paper_algorithms(performance_for_rho, rho)
            written.append(
                draw_performance_plot(
                    performance_for_rho,
                    rho,
                    output_dir,
                )
            )
            written.append(
                draw_residual_plot(
                    residual_for_rho,
                    rho,
                    output_dir,
                    show_x_label=np.isclose(rho, middle_rho),
                    show_y_label=np.isclose(rho, first_rho),
                )
            )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("Created plots:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
