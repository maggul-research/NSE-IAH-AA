#!/usr/bin/env python3
"""Run the channel-flow parameter study reported in the paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IAH and AA-IAH for the full-step channel-flow benchmark. "
            "The defaults reproduce the paper's Re, rho, alpha, mesh, "
            "tolerance, and Anderson-depth sweep."
        )
    )
    parser.add_argument(
        "--iah-executable",
        type=Path,
        default=SCRIPT_DIR / "channel_IAH",
        help="path to channel_IAH (default: beside this script)",
    )
    parser.add_argument(
        "--aa-executable",
        type=Path,
        default=SCRIPT_DIR / "channel_AA_IAH",
        help="path to channel_AA_IAH (default: beside this script)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="root output directory (default: ./results)",
    )
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 200.0],
        help="rho values (default: 50 100 200)",
    )
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--refinements", type=int, default=4)
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
        help="replace known CSV and VTK outputs in each selected rho directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.re <= 0 or args.alpha <= 0 or args.tolerance <= 0:
        raise ValueError("--re, --alpha, and --tolerance must be positive")
    if args.refinements < 0 or args.max_iterations <= 0:
        raise ValueError("--refinements must be nonnegative and --max-iterations positive")
    if not args.rho or any(value <= 0 for value in args.rho):
        raise ValueError("all --rho values must be positive")
    if not args.m or any(value <= 0 for value in args.m):
        raise ValueError("all --m values must be positive")

    args.rho = list(dict.fromkeys(args.rho))
    args.m = list(dict.fromkeys(args.m))

    for executable in (args.iah_executable, args.aa_executable):
        if not executable.is_file():
            raise FileNotFoundError(f"Executable not found: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"File is not executable: {executable}")


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def clean_known_outputs(directory: Path) -> list[Path]:
    outputs = [
        directory / "performance_summary.csv",
        directory / "residue_history.csv",
        *directory.glob("Channel_Re_*_solution_*.vtk"),
    ]
    existing = [path for path in outputs if path.exists()]
    for path in existing:
        path.unlink()
    return existing


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"[{cwd}] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def common_options(args: argparse.Namespace, rho: float) -> list[str]:
    return [
        "--re",
        f"{args.re:g}",
        "--rho",
        f"{rho:g}",
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

    for rho in args.rho:
        output_dir = args.results_dir / f"rho_{number_token(rho)}"
        output_dir.mkdir(parents=True, exist_ok=True)

        known_outputs = [
            output_dir / "performance_summary.csv",
            output_dir / "residue_history.csv",
            *output_dir.glob("Channel_Re_*_solution_*.vtk"),
        ]
        existing = [path for path in known_outputs if path.exists()]
        if existing and not args.overwrite:
            print(
                f"error: {output_dir} already contains generated output; "
                "use --overwrite to replace it",
                file=sys.stderr,
            )
            return 2
        if args.overwrite and not args.dry_run:
            clean_known_outputs(output_dir)

        options = common_options(args, rho)
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

    print(
        "Run complete. Generate the paper plots with:\n"
        f"  {shlex.join([sys.executable, str(SCRIPT_DIR / 'channel_plots.py'), '--results-dir', str(args.results_dir)])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
