# Anderson Acceleration of the Improved Arrow–Hurwicz Method for the Steady-State Navier–Stokes Equations

> **Sinan Ergen · Mustafa Aggul · Mustafa Türkyılmazoğlu**  
> Balıkesir University · Southern Methodist University · Hacettepe University  
> © 2026

This repository contains the source code used to produce all numerical results reported in the paper. The simulations are implemented in C++ using the [deal.II](https://www.dealii.org/) finite element library, and post-processing is performed with Python using Matplotlib, pandas, and NumPy.

The reported results were produced with deal.II v9.8.0-pre (commit `82c56fc0ed`). The CMake superbuild reuses any compatible deal.II installation (v9.7.1 or newer). If no compatible installation is found, deal.II can optionally be initialized from the local submodule or downloaded and installed by configuring with `-DIAH_BOOTSTRAP_DEAL_II=ON`.

---

## Repository Structure

```text
.
├── accuracy/
│   ├── accuracy_IAH.cc                  # accuracy and convergence test for IAH
│   └── accuracy_AA_IAH.cc               # accuracy and convergence test for AA-IAH
│
├── lid_driven_cavity/
│   ├── cavity_IAH.cc                    # lid-driven cavity simulation with IAH
│   ├── cavity_AA_IAH.cc                 # lid-driven cavity simulation with AA-IAH
│   ├── run_cavity.py                    # runs the complete paper parameter study
│   ├── cavity_performance_plots.py      # post-processing plots for performance
│   └── cavity_comparison_plots.py       # post-processing plots for reference-data comparison
│
├── channel_flow/
│   ├── channel_IAH.cc                   # channel-flow-over-a-step simulation with IAH
│   ├── channel_AA_IAH.cc                # channel-flow-over-a-step simulation with AA-IAH
│   ├── run_channel.py                   # runs the complete paper parameter study
│   └── channel_plots.py                 # reads all runs and creates the paper plots
│
├── dealii/                              # optional deal.II submodule
├── CMakeLists.txt                       # CMake superbuild
└── README.md                            # project documentation
```

---

## File Descriptions

### C++ Source Files and Post-Processing Scripts

| File | Description |
|------|-------------|
| `accuracy/accuracy_IAH.cc` | Implements the classical IAH method on a manufactured solution on the unit square $\Omega = (0,1)^2$. Performs successive mesh refinements $(N=2,\ldots,8)$ and reports velocity and pressure errors in multiple norms to verify theoretical convergence rates. This implementation as hard-coded parameters produces the Table 1 results in the paper. Since the accuracy results are identical for both IAH and AA-IAH when the convergence is achieved, only the IAH results are provided for reference. |
| `accuracy/accuracy_AA_IAH.cc` | Performs the same convergence test with Anderson Acceleration (AA-IAH) at depth $m=2$. Because AA does not alter the spatial discretization, the errors match those of the IAH code. This implementation as hard-coded parameters produces the Table 2 results in the paper. Since the accuracy results are identical for both IAH and AA-IAH when the convergence is achieved, only the AA-IAH results are provided for reference. |
| `lid_driven_cavity/cavity_IAH.cc` | Solves the lid-driven cavity problem using IAH. Accepts terminal parameters and writes residual, performance, VTK, and centerline-profile outputs. |
| `lid_driven_cavity/cavity_AA_IAH.cc` | Solves the same problem with Anderson Acceleration. Accepts terminal parameters, including the memory depth $m$, and writes the same result types. |
| `lid_driven_cavity/run_cavity.py` | Runs the six paper Reynolds numbers for IAH and AA-IAH($m=1,\ldots,5$), placing each Reynolds-number study in a separate result directory. |
| `lid_driven_cavity/cavity_performance_plots.py` | Reads all cavity result directories and produces the six performance panels and six residual-history panels used in the paper. The residual panels compare IAH with AA-IAH($m=4$), as specified in the manuscript. |
| `lid_driven_cavity/cavity_comparison_plots.py` | Reads the generated centerline CSV files and produces the $Re=10{,}000$, $12{,}500$, and $15{,}000$ velocity-profile panels with zoomed insets used in the paper. |
| `channel_flow/channel_IAH.cc` | Solves channel flow over a full step $(Re=100, 480\times160\text{ resolution}, 692{,}179\text{ DoFs})$ using the IAH method. Outputs residual history and a performance summary. |
| `channel_flow/channel_AA_IAH.cc` | Solves the same channel-flow problem with Anderson Acceleration (AA-IAH), using $m=1,\ldots,5$. |
| `channel_flow/run_channel.py` | Runs IAH and AA-IAH for $\rho=50,100,200$ and $m=1,\ldots,5$, placing each $\rho$ sweep in a separate result directory. |
| `channel_flow/channel_plots.py` | Reads all channel-flow result directories and produces the three performance panels and three residual-history panels used in the paper. The residual panels compare IAH with AA-IAH($m=4$), as specified in the manuscript. |

Implementation details and parameter choices are provided in the manuscript.

Every C++ program writes its output into the current working directory. The cavity and channel-flow Python scripts accept explicit input and output directories as shown below.

---

## Requirements

- CMake 3.20 or newer
- Git
- A C++17 compiler
- deal.II v9.7.1 or newer, either:
  - installed separately, or
  - built by the optional superbuild
- Python with Matplotlib, pandas, and NumPy for post-processing

Release mode is strongly recommended for meaningful CPU-time comparisons.

---

## Compile and Run

Clone the repository:

```bash
git clone git@github.com:maggul-research/NSE-IAH-AA.git
cd NSE-IAH-AA
```

### Use an Existing deal.II Installation

The preferred approach is to provide the deal.II installation prefix explicitly:

```bash
cmake -S . -B build \
  -DDEAL_II_ROOT=/path/to/dealii-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The legacy `DEAL_II_DIR` environment-variable hint is also supported:

```bash
export DEAL_II_DIR=/path/to/dealii-install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Alternatively, use CMake's package-specific variable and point it directly to the directory containing `deal.IIConfig.cmake`:

```bash
cmake -S . -B build \
  -Ddeal.II_DIR=/path/to/dealii-install/lib/cmake/deal.II \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Depending on how deal.II was installed, `deal.IIConfig.cmake` may instead be under a location such as `share/deal.II`.

### Build deal.II Automatically

Automatic bootstrapping is opt-in. If no compatible installation is available, configure with:

```bash
cmake -S . -B build \
  -DIAH_BOOTSTRAP_DEAL_II=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

In bootstrap mode, CMake:

1. Uses an already initialized `dealii/` source tree, if present.
2. Otherwise initializes the `dealii/` Git submodule, if configured.
3. If no submodule is available, downloads the configured deal.II release.
4. Builds and installs deal.II under `build/_deps/dealii-install`.
5. Configures and builds one executable for every `.cc` source file.

The deal.II build is tracked by CMake and is not repeated on subsequent builds unless its configuration or source changes, or the build directory is removed. Application-source changes are rebuilt incrementally.

To ignore an installed deal.II package and force the bundled build path, use:

```bash
cmake -S . -B build \
  -DIAH_FORCE_BOOTSTRAP_DEAL_II=ON \
  -DCMAKE_BUILD_TYPE=Release
```

### MPI and p4est

The bootstrapped deal.II build disables MPI and p4est by default. Enable both when required:

```bash
cmake -S . -B build \
  -DIAH_BOOTSTRAP_DEAL_II=ON \
  -DIAH_DEAL_II_WITH_MPI=ON \
  -DIAH_DEAL_II_WITH_P4EST=ON \
  -DCMAKE_BUILD_TYPE=Release
```

`IAH_DEAL_II_WITH_P4EST=ON` requires `IAH_DEAL_II_WITH_MPI=ON`.

---

## Build Output

The outer superbuild and inner application build use separate directories. Executables are placed under `build/programs/` in directories corresponding to their source locations:

```text
build/programs/
├── accuracy/
│   ├── accuracy_IAH
│   └── accuracy_AA_IAH
├── lid_driven_cavity/
│   ├── cavity_IAH
│   ├── cavity_AA_IAH
│   ├── run_cavity.py
│   ├── cavity_performance_plots.py
│   └── cavity_comparison_plots.py
└── channel_flow/
    ├── channel_IAH
    ├── channel_AA_IAH
    ├── run_channel.py
    └── channel_plots.py
```

All `.py` files are copied automatically from each source directory into the corresponding build-output directory. Modified scripts are refreshed during CMake reconfiguration, and scripts added to or removed from the source tree are detected automatically.

For example:

```bash
cd build/programs/accuracy
./accuracy_IAH
```

For the full lid-driven cavity study reported in the paper:

```bash
cd build/programs/lid_driven_cavity
python3 run_cavity.py
python3 cavity_performance_plots.py --results-dir results --output-dir plots
python3 cavity_comparison_plots.py --results-dir results --output-dir plots
```

The cavity defaults are
$Re\in\{1000,2500,5000,10000,12500,15000\}$, $\rho=100$,
$\alpha=1$, $N=8$ (a $256\times256$ mesh with $592{,}387$ DoFs),
tolerance $10^{-6}$, and $m\in\{1,2,3,4,5\}$. The full sweep is
computationally expensive. Existing generated output is protected; pass
`--overwrite` to `run_cavity.py` when intentionally replacing a previous
sweep.

Both cavity executables also accept terminal parameters directly. For example:

```bash
./cavity_IAH --re 10000 --rho 100 --alpha 1 --refinements 8
./cavity_AA_IAH --re 10000 --rho 100 --alpha 1 --m 4 --refinements 8
```

Use `--help` on either executable or Python script for the complete option
list. With no `--m` argument, `cavity_AA_IAH` runs all five paper depths.

For the full channel-flow study reported in the paper:

```bash
cd build/programs/channel_flow
python3 run_channel.py
python3 channel_plots.py --results-dir results --output-dir plots
```

The paper defaults are $Re=100$, $\alpha=100$, $N=4$, tolerance $10^{-6}$, $\rho\in\{50,100,200\}$, and $m\in\{1,2,3,4,5\}$. The full sweep is computationally expensive. Existing generated output is protected; pass `--overwrite` to `run_channel.py` when intentionally replacing a previous sweep.

Both channel executables also accept terminal parameters directly. For example:

```bash
./channel_IAH --re 100 --rho 50 --alpha 100 --refinements 4
./channel_AA_IAH --re 100 --rho 50 --alpha 100 --m 5 --refinements 4
```

Use `--help` on either executable or Python script for the complete option list. With no `--m` argument, `channel_AA_IAH` runs all five paper depths.

---

## Notes

- The cavity and channel programs create CSV headers when needed and then append, so IAH and AA-IAH may be run in either order. Their CSV files store $Re$, $\rho$, $\alpha$, $m$, and the refinement level.
- The lid-driven cavity case writes `Performance_Summary.csv` and `Residual_History.csv`; the channel-flow case writes `performance_summary.csv` and `residue_history.csv`. These names are case-sensitive.
- In the accuracy codes, $\theta$ is hard-coded in `ExactSolution`, `RightHandSide`, and `ExactVelocityGradient`. All three must be changed together.
- The cavity executables sample the converged finite-element solution directly on the vertical and horizontal centerlines and write `centerline_Re_..._x.csv` and `centerline_Re_..._y.csv`. This export occurs after the solver timer stops, so it is not included in the reported CPU time.

---

## Citation

If you use this code, please cite:

```text
Sinan Ergen, Mustafa Aggul, Mustafa Türkyılmazoğlu.
"Accelerating the Improved Arrow–Hurwicz Iteration via the Anderson Algorithm
for Steady-State Navier–Stokes Equations." (2026)
```

---

## License

© 2026 Mustafa Aggul and Sinan Ergen. All rights reserved.
