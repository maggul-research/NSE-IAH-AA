#!/usr/bin/env python3
"""Create the cavity performance and residual plots used in the paper."""

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

PAPER_REYNOLDS_NUMBERS = [1000, 2500, 5000, 10000, 12500, 15000]
ALL_ALGORITHMS = [
    "IAH",
    "AA-IAH(m=1)",
    "AA-IAH(m=2)",
    "AA-IAH(m=3)",
    "AA-IAH(m=4)",
    "AA-IAH(m=5)",
]
TARGET_AA_ALGORITHM = "AA-IAH(m=4)"

TITLE_FS = 46
LABEL_FS = 46
TICK_FS = 38
BAR_LABEL_FS = 36
XTICK_ALGO_FS = 42
RESID_TICK_FS = 42
RESID_LEGEND_FS = 40

BAR_FIG_W, BAR_FIG_H = 10, 6
# The residual panels use oversized labels and titles; give them additional
# horizontal and vertical canvas space so neither is clipped in the PDF.
RESID_FIG_W, RESID_FIG_H = 12, 8
BAR_MARGINS = dict(left=0.05, right=0.95, top=0.88, bottom=0.10)
RESID_MARGINS = dict(left=0.24, right=0.96, top=0.86, bottom=0.18)

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
            "Read every re_* result directory produced by run_cavity.py and "
            "write the performance and residual PDF panels used in the paper."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory containing re_*/ CSV results (default: ./results)",
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
        nargs="+",
        default=[float(value) for value in PAPER_REYNOLDS_NUMBERS],
        help="Reynolds numbers to plot (default: the six paper values)",
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
        total_minutes = int(round(total_seconds / 60))
        return f"{total_minutes // 60}h:{total_minutes % 60}m"
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


def algorithm_from_depth(depth: float) -> str:
    memory_depth = int(depth)
    return "IAH" if memory_depth == 0 else f"AA-IAH(m={memory_depth})"


def normalize_algorithms(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "Algorithm" not in data.columns:
        if "m" not in data.columns:
            raise ValueError("CSV has neither an Algorithm column nor an m column")
        data["Algorithm"] = data["m"].apply(algorithm_from_depth)
    data["Algorithm"] = (
        data["Algorithm"]
        .astype(str)
        .str.replace("AH-GAH", "AA-IAH", regex=False)
        .replace("GAH", "IAH")
    )
    return data


def csv_pairs(results_dir: Path) -> list[tuple[Path, Path]]:
    nested_performance = sorted(results_dir.glob("re_*/Performance_Summary.csv"))
    if nested_performance:
        pairs = []
        for performance_path in nested_performance:
            residual_path = performance_path.with_name("Residual_History.csv")
            if not residual_path.is_file():
                raise FileNotFoundError(f"Missing matching residual file: {residual_path}")
            pairs.append((performance_path, residual_path))
        return pairs

    performance_path = results_dir / "Performance_Summary.csv"
    residual_path = results_dir / "Residual_History.csv"
    if performance_path.is_file() and residual_path.is_file():
        return [(performance_path, residual_path)]

    raise FileNotFoundError(
        f"No re_*/Performance_Summary.csv files found under {results_dir}"
    )


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []

    for performance_path, residual_path in csv_pairs(results_dir):
        performance_frames.append(normalize_algorithms(pd.read_csv(performance_path)))
        residual_frames.append(normalize_algorithms(pd.read_csv(residual_path)))

    performance = pd.concat(performance_frames, ignore_index=True)
    residual = pd.concat(residual_frames, ignore_index=True)

    required_performance = {
        "Re",
        "Algorithm",
        "TotalIterations",
        "CPUTime",
    }
    required_residual = {"Re", "Algorithm", "Iteration", "Residual"}
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

    for data, columns in (
        (performance, ["Re", "TotalIterations", "CPUTime"]),
        (residual, ["Re", "Iteration", "Residual"]),
    ):
        for column in columns:
            data[column] = pd.to_numeric(data[column], errors="raise")

    return performance, residual


def select_reynolds_rows(data: pd.DataFrame, reynolds_number: float) -> pd.DataFrame:
    return data[np.isclose(data["Re"].astype(float), reynolds_number)].copy()


def verify_paper_runs(performance: pd.DataFrame, reynolds_number: float) -> None:
    counts = performance["Algorithm"].astype(str).value_counts()
    missing = [algorithm for algorithm in ALL_ALGORITHMS if algorithm not in counts]
    if missing:
        raise ValueError(
            f"Re={reynolds_number:g} is missing paper runs: {', '.join(missing)}"
        )
    repeated = [algorithm for algorithm in ALL_ALGORITHMS if counts[algorithm] != 1]
    if repeated:
        raise ValueError(
            f"Re={reynolds_number:g} has repeated runs: {', '.join(repeated)}; "
            "rerun with --overwrite"
        )


def draw_performance_plot(
    performance: pd.DataFrame,
    reynolds_number: float,
    output_dir: Path,
) -> Path:
    data = performance[performance["Algorithm"].isin(ALL_ALGORITHMS)].copy()
    data["Algorithm"] = pd.Categorical(
        data["Algorithm"], categories=ALL_ALGORITHMS, ordered=True
    )
    data = data.sort_values("Algorithm")

    figure, left_axis = plt.subplots(figsize=(BAR_FIG_W, BAR_FIG_H))
    left_axis.grid(
        True, axis="both", linestyle="-", color="lightgray", alpha=0.7, zorder=0
    )

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
    left_axis.set_title(f"Re = {int(reynolds_number)}", fontsize=TITLE_FS)
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
                fontsize=BAR_LABEL_FS,
                rotation=90,
                fontweight="normal",
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
                fontsize=BAR_LABEL_FS,
                rotation=90,
                fontweight="normal",
                clip_on=True,
            )

    for axis in (left_axis, right_axis):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    figure.subplots_adjust(**BAR_MARGINS)
    output_path = output_dir / f"cavity_performance_re_{int(reynolds_number)}.pdf"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def draw_residual_plot(
    residual: pd.DataFrame,
    reynolds_number: float,
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(RESID_FIG_W, RESID_FIG_H))
    axis.grid(True, which="both", linestyle="-", color="lightgray", alpha=0.7)

    for algorithm, color in (
        ("IAH", COLOR_IAH_LINE),
        (TARGET_AA_ALGORITHM, COLOR_AA_LINE),
    ):
        data = residual[residual["Algorithm"] == algorithm].sort_values("Iteration")
        if data.empty:
            raise ValueError(
                f"Re={reynolds_number:g} has no residual history for {algorithm}"
            )
        axis.plot(
            data["Iteration"],
            data["Residual"],
            label="IAH" if algorithm == "IAH" else "AA-IAH",
            color=color,
            linewidth=3.5,
            zorder=5,
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation_formatter))
    axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=20))
    axis.set_ylim(bottom=RESID_YMIN)

    axis.set_xlabel(
        "Iteration Count" if np.isclose(reynolds_number, 12500.0) else "",
        fontsize=LABEL_FS,
    )
    show_y_label = any(
        np.isclose(reynolds_number, value) for value in (1000.0, 10000.0)
    )
    axis.set_ylabel(
        "" if show_y_label else "", fontsize=LABEL_FS
    )
    axis.set_title(f"Re = {int(reynolds_number)}", fontsize=TITLE_FS)
    axis.tick_params(axis="both", labelsize=RESID_TICK_FS)
    if not show_y_label:
        axis.tick_params(axis="y", labelleft=False)

    if np.isclose(reynolds_number, 1000.0):
        axis.legend(
            fontsize=RESID_LEGEND_FS,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="black",
        )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    figure.subplots_adjust(**RESID_MARGINS)
    output_path = (
        output_dir / f"cavity_residual_history_re_{int(reynolds_number)}.pdf"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    try:
        performance, residual = load_results(args.results_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[Path] = []
        for reynolds_number in dict.fromkeys(args.re):
            performance_rows = select_reynolds_rows(performance, reynolds_number)
            residual_rows = select_reynolds_rows(residual, reynolds_number)
            if performance_rows.empty:
                raise ValueError(f"No performance data found for Re={reynolds_number:g}")
            if residual_rows.empty:
                raise ValueError(f"No residual data found for Re={reynolds_number:g}")
            verify_paper_runs(performance_rows, reynolds_number)
            output_paths.append(
                draw_performance_plot(
                    performance_rows, reynolds_number, args.output_dir
                )
            )
            output_paths.append(
                draw_residual_plot(residual_rows, reynolds_number, args.output_dir)
            )
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Created {len(output_paths)} paper plot files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
