#!/usr/bin/env python3
"""Compare cavity centerline profiles with Erturk et al. (2005)."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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


# Reference data reported by Erturk et al. (2005).
y_ref = np.array(
    [
        1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90,
        0.50, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00,
    ]
)
x_ref = np.array(
    [
        1.000, 0.985, 0.970, 0.955, 0.940, 0.925, 0.910, 0.895, 0.880,
        0.865, 0.850, 0.500, 0.150, 0.135, 0.120, 0.105, 0.090, 0.075,
        0.060, 0.045, 0.030, 0.015, 0.000,
    ]
)

ref_ux = {
    1000: np.array([1.0000, 0.8486, 0.7065, 0.5917, 0.5102, 0.4582, 0.4276, 0.4101, 0.3993, 0.3913, 0.3838, -0.0620, -0.3756, -0.3869, -0.3854, -0.3690, -0.3381, -0.2960, -0.2472, -0.1951, -0.1392, -0.0757, 0.0000]),
    2500: np.array([1.0000, 0.7704, 0.5924, 0.4971, 0.4607, 0.4506, 0.4470, 0.4424, 0.4353, 0.4256, 0.4141, -0.0403, -0.3228, -0.3439, -0.3688, -0.3965, -0.4200, -0.4250, -0.3979, -0.3372, -0.2547, -0.1517, 0.0000]),
    5000: np.array([1.0000, 0.6866, 0.5159, 0.4749, 0.4739, 0.4738, 0.4683, 0.4582, 0.4452, 0.4307, 0.4155, -0.0319, -0.3100, -0.3285, -0.3467, -0.3652, -0.3876, -0.4168, -0.4419, -0.4272, -0.3480, -0.2223, 0.0000]),
    10000: np.array([1.0000, 0.5891, 0.4837, 0.4891, 0.4917, 0.4843, 0.4711, 0.4556, 0.4398, 0.4243, 0.4095, -0.0268, -0.2998, -0.3179, -0.3361, -0.3543, -0.3721, -0.3899, -0.4142, -0.4469, -0.4259, -0.2907, 0.0000]),
    12500: np.array([1.0000, 0.5587, 0.4833, 0.4941, 0.4937, 0.4833, 0.4684, 0.4523, 0.4366, 0.4216, 0.4070, -0.0256, -0.2967, -0.3146, -0.3326, -0.3506, -0.3685, -0.3859, -0.4054, -0.4380, -0.4407, -0.3113, 0.0000]),
    15000: np.array([1.0000, 0.5358, 0.4850, 0.4969, 0.4937, 0.4811, 0.4653, 0.4492, 0.4338, 0.4190, 0.4047, -0.0247, -0.2942, -0.3119, -0.3297, -0.3474, -0.3652, -0.3827, -0.4001, -0.4286, -0.4474, -0.3278, 0.0000]),
    20000: np.array([1.0000, 0.5048, 0.4889, 0.4985, 0.4906, 0.4754, 0.4592, 0.4436, 0.4287, 0.4142, 0.4001, -0.0234, -0.2899, -0.3074, -0.3248, -0.3422, -0.3595, -0.3769, -0.3936, -0.4143, -0.4475, -0.3523, 0.0000]),
}

ref_uy = {
    1000: np.array([0.0000, -0.0973, -0.2173, -0.3400, -0.4417, -0.5052, -0.5263, -0.5132, -0.4803, -0.4407, -0.4028, 0.0258, 0.3756, 0.3705, 0.3605, 0.3460, 0.3273, 0.3041, 0.2746, 0.2349, 0.1792, 0.1019, 0.0000]),
    2500: np.array([0.0000, -0.1675, -0.3725, -0.5192, -0.5603, -0.5268, -0.4741, -0.4321, -0.4042, -0.3843, -0.3671, 0.0160, 0.3918, 0.4078, 0.4187, 0.4217, 0.4142, 0.3950, 0.3649, 0.3238, 0.2633, 0.1607, 0.0000]),
    5000: np.array([0.0000, -0.2441, -0.5019, -0.5700, -0.5139, -0.4595, -0.4318, -0.4147, -0.3982, -0.3806, -0.3624, 0.0117, 0.3699, 0.3878, 0.4070, 0.4260, 0.4403, 0.4426, 0.4258, 0.3868, 0.3263, 0.2160, 0.0000]),
    10000: np.array([0.0000, -0.3419, -0.5712, -0.5124, -0.4592, -0.4411, -0.4256, -0.4078, -0.3895, -0.3715, -0.3538, 0.0088, 0.3562, 0.3722, 0.3885, 0.4056, 0.4247, 0.4449, 0.4566, 0.4409, 0.3844, 0.2756, 0.0000]),
    12500: np.array([0.0000, -0.3762, -0.5694, -0.4899, -0.4534, -0.4388, -0.4221, -0.4040, -0.3859, -0.3682, -0.3508, 0.0080, 0.3519, 0.3678, 0.3840, 0.4004, 0.4180, 0.4383, 0.4563, 0.4522, 0.4018, 0.2940, 0.0000]),
    15000: np.array([0.0000, -0.4041, -0.5593, -0.4754, -0.4505, -0.4361, -0.4186, -0.4005, -0.3828, -0.3654, -0.3481, 0.0074, 0.3483, 0.3641, 0.3801, 0.3964, 0.4132, 0.4323, 0.4529, 0.4580, 0.4152, 0.3083, 0.0000]),
    20000: np.array([0.0000, -0.4457, -0.5321, -0.4605, -0.4459, -0.4300, -0.4122, -0.3946, -0.3774, -0.3603, -0.3434, 0.0065, 0.3423, 0.3579, 0.3736, 0.3897, 0.4060, 0.4232, 0.4438, 0.4601, 0.4332, 0.3290, 0.0000]),
}


PAPER_REYNOLDS_NUMBERS = [10000, 12500, 15000]
MODELS = ["IAH", "m1", "m2", "m3", "m4", "m5"]
MODEL_LABELS = {
    "IAH": "Classic IAH",
    "m1": "m=1",
    "m2": "m=2",
    "m3": "m=3",
    "m4": "m=4",
    "m5": "m=5",
}
COLORS = {
    "IAH": "#000000",
    "m1": "#3498db",
    "m2": "#e67e22",
    "m3": "#2ecc71",
    "m4": "#9b59b6",
    "m5": "#e74c3c",
}
LINESTYLES = {"IAH": "--", "m1": "-", "m2": "-", "m3": "-", "m4": "-", "m5": "-"}
LINEWIDTHS_MAIN = {"IAH": 4.0, "m1": 3.2, "m2": 3.2, "m3": 3.2, "m4": 3.2, "m5": 3.2}

TITLE_FS = 42
YLABEL_FS = 42
XLABEL_FS = 42
TICK_FS = 40
LEGEND_FS = 38
INSET_TICK_FS = 28
ZOOM_LABEL_FS = 26
# Extra left-side canvas space keeps the large velocity-axis label fully inside
# the PDF without changing the right, top, or bottom boundaries.
PROFILE_MARGINS = dict(left=0.22, right=0.97, top=0.90, bottom=0.16)


def draw_shared_legend(output_dir: Path) -> Path:
    """Create the common profile legend as a standalone PDF."""
    handles = [
        Line2D(
            [], [], marker="o", linestyle="None", color="black", markersize=9,
            label="Reference",
        )
    ]
    handles.extend(
        Line2D(
            [], [],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=LINEWIDTHS_MAIN[model],
            alpha=0.85,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    )

    figure = plt.figure(figsize=(24.0, 2.2))
    figure.legend(
        handles=handles,
        ncol=len(handles),
        fontsize=LEGEND_FS,
        loc="center",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="black",
    )
    output_path = output_dir / "cavity_centerline_legend.pdf"
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the native centerline CSV files produced by the cavity "
            "executables and create the paper's comparison panels."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory containing re_*/ centerline files (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../../plots"),
        help="plot output directory (default: plots beside the build directory)",
    )
    parser.add_argument(
        "--re",
        type=int,
        nargs="+",
        default=PAPER_REYNOLDS_NUMBERS,
        help="Reynolds numbers (default: 10000 12500 15000, as in the paper)",
    )
    parser.add_argument(
        "--legacy-xyz-dir",
        type=Path,
        help="optional directory containing the original VisIt .xyz files",
    )
    return parser.parse_args()


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def read_centerline_csv(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename)
    required = {"Coordinate", "Velocity"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {', '.join(sorted(missing))}"
        )
    data = data[["Coordinate", "Velocity"]].apply(pd.to_numeric, errors="raise")
    data = data.sort_values("Coordinate").drop_duplicates(subset="Coordinate")
    if data.empty:
        raise ValueError(f"{filename} contains no centerline samples")
    return data["Coordinate"].to_numpy(), data["Velocity"].to_numpy()


def read_visit_file(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filename, sep="\t", header=None, skiprows=2).iloc[:, [1, 4]]
    data.columns = ["Coordinate", "Velocity"]
    data = data.apply(pd.to_numeric, errors="raise")
    data = data.sort_values("Coordinate").drop_duplicates(subset="Coordinate")
    return data["Coordinate"].to_numpy(), data["Velocity"].to_numpy()


def model_depth(model: str) -> int:
    return 0 if model == "IAH" else int(model.removeprefix("m"))


def native_profile_path(
    results_dir: Path,
    reynolds_number: int,
    model: str,
    component: str,
) -> Path | None:
    depth = model_depth(model)
    search_directories = [
        results_dir / f"re_{number_token(float(reynolds_number))}",
        results_dir,
    ]
    matches: list[Path] = []
    pattern = re.compile(r"^centerline_Re_(.+?)_rho_.+_m_\d+_[xy]\.csv$")
    for directory in search_directories:
        if not directory.is_dir():
            continue
        for candidate in directory.glob(f"centerline_Re_*_rho_*_m_{depth}_{component}.csv"):
            match = pattern.match(candidate.name)
            if match and np.isclose(float(match.group(1)), reynolds_number):
                matches.append(candidate)

    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        raise ValueError(
            f"Multiple centerline files match Re={reynolds_number}, {model}, {component}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def load_profile(
    results_dir: Path,
    legacy_xyz_dir: Path | None,
    reynolds_number: int,
    model: str,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    native_path = native_profile_path(
        results_dir, reynolds_number, model, component
    )
    if native_path is not None:
        return read_centerline_csv(native_path)

    if legacy_xyz_dir is not None:
        suffix = "IAH" if model == "IAH" else model
        legacy_path = legacy_xyz_dir / f"{component}{reynolds_number}{suffix}.xyz"
        if legacy_path.is_file():
            return read_visit_file(legacy_path)

    raise FileNotFoundError(
        "No centerline data found for "
        f"Re={reynolds_number}, {model}, component={component}. "
        "Run run_cavity.py, or provide --legacy-xyz-dir."
    )


def draw_profile(
    results_dir: Path,
    output_dir: Path,
    legacy_xyz_dir: Path | None,
    reynolds_number: int,
    component: str,
) -> Path:
    if component == "x":
        reference_coordinate = y_ref
        reference_velocity = ref_ux[reynolds_number]
        x_label = r"u-Velocity ($u_x$)"
        y_label = "y-Coordinate"
    else:
        reference_coordinate = x_ref
        reference_velocity = ref_uy[reynolds_number]
        x_label = "x-Coordinate"
        y_label = r"v-Velocity ($u_y$)"

    figure, axis = plt.subplots(figsize=(12, 10))
    if component == "x":
        axis.plot(
            reference_velocity,
            reference_coordinate,
            "o",
            label="Reference",
            color="black",
            markersize=9,
            zorder=10,
        )
    else:
        axis.plot(
            reference_coordinate,
            reference_velocity,
            "o",
            label="Reference",
            color="black",
            markersize=6,
            zorder=10,
        )

    enable_zoom = reynolds_number >= 10000
    inset = None
    if enable_zoom:
        if component == "x":
            inset = axis.inset_axes([0.54, 0.18, 0.42, 0.40])
            coordinate_limits = (0.0592, 0.0608)
        else:
            inset = axis.inset_axes([0.62, 0.65, 0.36, 0.32])
            coordinate_limits = (0.9698, 0.9701)
        inset.set_facecolor("#ffffff")
        inset.grid(True, linestyle="--", alpha=0.5)

    window_values: list[float] = []
    for model in MODELS:
        coordinates, velocities = load_profile(
            results_dir,
            legacy_xyz_dir,
            reynolds_number,
            model,
            component,
        )
        if component == "x":
            axis.plot(
                velocities,
                coordinates,
                label=MODEL_LABELS[model],
                color=COLORS[model],
                linestyle=LINESTYLES[model],
                linewidth=LINEWIDTHS_MAIN[model],
                alpha=0.85,
            )
            if inset is not None:
                inset.plot(
                    velocities,
                    coordinates,
                    color=COLORS[model],
                    linestyle=LINESTYLES[model],
                    linewidth=2.8,
                )
        else:
            axis.plot(
                coordinates,
                velocities,
                label=MODEL_LABELS[model],
                color=COLORS[model],
                linestyle=LINESTYLES[model],
                linewidth=LINEWIDTHS_MAIN[model],
                alpha=0.85,
            )
            if inset is not None:
                inset.plot(
                    coordinates,
                    velocities,
                    color=COLORS[model],
                    linestyle=LINESTYLES[model],
                    linewidth=2.8,
                )

        if inset is not None:
            mask = (coordinates >= coordinate_limits[0]) & (
                coordinates <= coordinate_limits[1]
            )
            window_values.extend(velocities[mask].tolist())

    if inset is not None and window_values:
        if component == "x":
            reference_points = [ref_ux[reynolds_number][20], ref_ux[reynolds_number][19]]
            velocity_limits = (
                min(window_values + reference_points) - 0.0002,
                max(window_values + reference_points) + 0.0002,
            )
            inset.set_xlim(velocity_limits)
            inset.set_ylim(coordinate_limits)
            inset.plot(
                reference_velocity,
                reference_coordinate,
                "o",
                color="black",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=10,
            )
        else:
            reference_points = [ref_uy[reynolds_number][2]]
            velocity_limits = (
                min(window_values + reference_points) - 0.0002,
                max(window_values + reference_points) + 0.0002,
            )
            inset.set_xlim(coordinate_limits)
            inset.set_ylim(velocity_limits)
            inset.plot(
                reference_coordinate,
                reference_velocity,
                "o",
                color="black",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=10,
            )

        inset.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
        inset.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
        inset.tick_params(axis="both", which="major", labelsize=INSET_TICK_FS)
        inset.ticklabel_format(axis="both", style="plain", useOffset=False)
        plt.setp(
            inset.get_xticklabels(),
            rotation=25,
            ha="right",
            rotation_mode="anchor",
        )
        axis.indicate_inset_zoom(inset, edgecolor="black", alpha=0.5, linewidth=2.0)
        for spine in inset.spines.values():
            spine.set_linewidth(2.0)

    axis.set_xlabel(x_label if reynolds_number == 15000 else "", fontsize=XLABEL_FS)
    axis.set_ylabel(y_label, fontsize=YLABEL_FS)
    axis.set_title(f"Re = {reynolds_number}", fontsize=TITLE_FS)
    axis.tick_params(axis="both", which="major", labelsize=TICK_FS)
    axis.grid(True, linestyle="--", alpha=0.6)

    figure.subplots_adjust(**PROFILE_MARGINS)
    velocity_component = "ux" if component == "x" else "uy"
    output_path = (
        output_dir
        / f"cavity_centerline_{velocity_component}_re_{reynolds_number}.pdf"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.legacy_xyz_dir is not None:
        args.legacy_xyz_dir = args.legacy_xyz_dir.expanduser().resolve()

    try:
        reynolds_numbers = list(dict.fromkeys(args.re))
        unsupported = [
            value for value in reynolds_numbers if value not in ref_ux or value not in ref_uy
        ]
        if unsupported:
            raise ValueError(
                "No embedded reference data for Re="
                + ", ".join(str(value) for value in unsupported)
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [
            draw_profile(
                args.results_dir,
                args.output_dir,
                args.legacy_xyz_dir,
                reynolds_number,
                component,
            )
            for reynolds_number in reynolds_numbers
            for component in ("x", "y")
        ]
        output_paths.append(draw_shared_legend(args.output_dir))
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Created {len(output_paths)} centerline comparison files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
