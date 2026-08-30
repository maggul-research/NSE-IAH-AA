#!/usr/bin/env python3
"""Run the lid-driven-cavity parameter study reported in the paper."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_REYNOLDS_NUMBERS = [1000.0, 2500.0, 5000.0, 10000.0, 12500.0, 15000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IAH and AA-IAH for the lid-driven cavity. The defaults "
            "reproduce the paper's Reynolds-number, rho, alpha, mesh, "
            "tolerance, and Anderson-depth sweep."
        )
    )
    parser.add_argument(
        "--iah-executable",
        type=Path,
        default=SCRIPT_DIR / "cavity_IAH",
        help="path to cavity_IAH (default: beside this script)",
    )
    parser.add_argument(
        "--aa-executable",
        type=Path,
        default=SCRIPT_DIR / "cavity_AA_IAH",
        help="path to cavity_AA_IAH (default: beside this script)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="root output directory (default: ./results)",
    )
    parser.add_argument(
        "--re",
        type=float,
        nargs="+",
        default=PAPER_REYNOLDS_NUMBERS,
        help="Reynolds numbers (default: 1000 2500 5000 10000 12500 15000)",
    )
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--refinements", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Anderson depths (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known CSV, VTK, and centerline outputs in selected directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    scalar_values = (args.rho, args.alpha, args.tolerance)
    if any(not math.isfinite(value) or value <= 0 for value in scalar_values):
        raise ValueError("--rho, --alpha, and --tolerance must be finite and positive")
    if args.refinements < 0 or args.max_iterations <= 0:
        raise ValueError("--refinements must be nonnegative and --max-iterations positive")
    if not args.re or any(not math.isfinite(value) or value <= 0 for value in args.re):
        raise ValueError("all --re values must be finite and positive")
    if not args.m or any(value <= 0 for value in args.m):
        raise ValueError("all --m values must be positive")

    args.re = list(dict.fromkeys(args.re))
    args.m = list(dict.fromkeys(args.m))

    for executable in (args.iah_executable, args.aa_executable):
        if not executable.is_file():
            raise FileNotFoundError(f"Executable not found: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"File is not executable: {executable}")


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def known_outputs(directory: Path) -> list[Path]:
    candidates = [
        directory / "Performance_Summary.csv",
        directory / "Residual_History.csv",
        *directory.glob("Cavity_Re_*_solution_*.vtk"),
        *directory.glob("centerline_Re_*.csv"),
    ]
    return [path for path in candidates if path.exists()]


def clean_known_outputs(directory: Path) -> None:
    for path in known_outputs(directory):
        path.unlink()


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"[{cwd}] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def common_options(args: argparse.Namespace, reynolds_number: float) -> list[str]:
    return [
        "--re",
        f"{reynolds_number:g}",
        "--rho",
        f"{args.rho:g}",
        "--alpha",
        f"{args.alpha:g}",
        "--refinements",
        str(args.refinements),
        "--tolerance",
        f"{args.tolerance:.17g}",
        "--max-iterations",
        str(args.max_iterations),
    ]


def main() -> int:
    args = parse_args()
    args.iah_executable = args.iah_executable.expanduser().resolve()
    args.aa_executable = args.aa_executable.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()

    try:
        validate(args)
    except (ValueError, FileNotFoundError, PermissionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for reynolds_number in args.re:
        output_dir = args.results_dir / f"re_{number_token(reynolds_number)}"
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = known_outputs(output_dir)
        if existing and not args.overwrite:
            print(
                f"error: {output_dir} already contains generated output; "
                "use --overwrite to replace it",
                file=sys.stderr,
            )
            return 2
        if args.overwrite and not args.dry_run:
            clean_known_outputs(output_dir)

        options = common_options(args, reynolds_number)
        run_command([str(args.iah_executable), *options], output_dir, args.dry_run)
        for memory_depth in args.m:
            run_command(
                [
                    str(args.aa_executable),
                    *options,
                    "--m",
                    str(memory_depth),
                ],
                output_dir,
                args.dry_run,
            )

    performance_script = SCRIPT_DIR / "cavity_performance_plots.py"
    comparison_script = SCRIPT_DIR / "cavity_comparison_plots.py"
    print(
        "Run complete. Generate the paper plots with:\n"
        f"  {shlex.join([sys.executable, str(performance_script), '--results-dir', str(args.results_dir)])}\n"
        f"  {shlex.join([sys.executable, str(comparison_script), '--results-dir', str(args.results_dir)])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
