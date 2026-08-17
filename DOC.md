# OpenModalPy — Technical Reference

This document is the single reference for humans and LLMs working with the
OpenModalPy codebase. It covers architecture, every supported method, the data
contract, the configuration system, the CLI, and extension paths.

---

## Architecture

```
src/openmodalpy/
├── __init__.py          # public exports: analyzers + set/get/blas_threads policy
├── core/
│   ├── base.py          # BaseAnalyzer, compute_reduced_svd, blocksfft,
│   │                    #   spod_function, weight calculation, plot helpers
│   ├── decomposition.py # lift / metric / weighted_second_order seam
│   │                    #   (POD, mPOD, ST-POD, PSD-POD share this)
│   ├── io.py            # MATDataLoader, DNamiDataLoader, _slice_block_in_time
│   ├── config.py        # FFT_BACKEND, FIG_DPI, directory defaults
│   ├── threads.py       # process-wide BLAS thread policy (default 1)
│   └── parallel.py      # thread-pool FFT + SPOD acceleration
├── pod.py               # PODAnalyzer (variance-optimal, identity lift)
├── mpod.py              # MPODAnalyzer (band-filtered POD)
├── psd_pod.py           # PSDPODAnalyzer (pooled Fourier-ensemble POD)
├── spod.py              # SPODAnalyzer (frequency-by-frequency POD)
├── dmd.py               # DMDAnalyzer (LS/TLS, delay embedding, HODMD)
├── bsmd.py              # BSMDAnalyzer (triadic bispectral decomposition)
├── stpod.py             # STPODAnalyzer (delay-embedded POD via Hankel lift)
├── commands.py          # dispatch core: analyze_from_spec, _run_pod_like,
│                        #   _run_dmd, _run_spod, _run_bsmd, _run_psd_pod,
│                        #   example discovery
├── cli.py               # argparse frontend: analyze, run, methods, examples, results
├── config_io.py         # load_jsonc, resolve_path, strip_jsonc_comments
├── specs.py             # DataSourceSpec, CaseSpec, AnalyzeSpec, RunOutcome, etc.
├── example_data.py      # built-in generators: double_gyre, taylor_green, cylinder_wake
└── examples/            # packaged .jsonc configs shipped in the wheel
```

FFT backend dispatch lives in the external [`fftkit`](https://github.com/openfluids/fftkit)
package: `get_fft_func()` selects among scipy/numpy/mkl/cupy/accelerate, and
`core.config.FFT_BACKEND` re-exports the backend fftkit resolved. Override it with the
`FFTKIT_BACKEND` environment variable.

## BLAS thread policy

OpenModalPy pins the process-wide BLAS/OpenMP thread count for `svd` / `eigh` /
`eig` so two runs on the same machine, same install, and same policy produce
**bit-identical** arrays by default. The default is **1 thread** (reproducible);
speed is opt-in.

```python
import openmodalpy as omp

omp.get_blas_threads()          # 1 unless OPENMODALPY_BLAS_THREADS is set
omp.set_blas_threads(4)         # process-wide; 0 means no limit from this package
with omp.blas_threads(0):       # scoped; restores the previous value
    ...
```

Environment variable `OPENMODALPY_BLAS_THREADS` is parsed lazily on the first
`get_blas_threads()` call (not at import time). `0` means this package applies
no limit — an existing `OMP_NUM_THREADS` or outer `threadpoolctl` limiter still
applies. The effective count is written into every result file as
`prov_blas_threads`.

**What is guaranteed.** Given a fixed environment (same OS, same NumPy/SciPy
build, same BLAS vendor and version, same policy), repeated runs are
deterministic: reduction order inside the kernels does not wander with core
count.

**What is not guaranteed.** Bit-identical results across BLAS vendors (OpenBLAS
vs MKL vs Accelerate) are not promised and are generally not achievable. Record
`prov_blas_threads` and the package versions in the provenance block when
comparing machines.


### Analyzer lifecycle

Every analyzer follows the same sequence:

1. **Construct** — pass `file_path`, loader, weight type, method params
2. **`load_and_preprocess()`** — load data → compute spatial weights → set derived params
3. **Method-specific computation** — `perform_pod()`, `perform_dmd()`, `perform_spod()`, etc.
4. **`save_results()`** — write HDF5 with modes, eigenvalues, metadata
5. **Plot** — `plot_eigenvalues()`, `plot_modes()`, etc.

The `commands.py` dispatch core (`analyze_from_spec`) automates steps 1–5 from
a single `AnalyzeSpec` dataclass, which is built from a JSONC config file.

### Design principle

All analyzers share:
- One **data contract** (snapshot matrix Q, coordinates, dt, spatial weights W)
- One **metric layer** (W defines the inner product for orthogonality and energy)
- One **lift** concept (the method-specific transformation of raw snapshots)

They differ only in the **operator problem** solved on the lifted data:
- Variance-optimal (eigendecomposition of weighted covariance/kernel)
- Evolution-fit (SVD-based regression on paired snapshots)
- Triadic interaction (cross-bispectral coupling optimization)

The variance-optimal path is a single seam in `core/decomposition.py`:
named lifts (`IdentityLift`, `DelayEmbeddingLift`, `BandFilteredLift`), a
`SpatialMetric` (with `.tile(d)` for delay space), and
`weighted_second_order(..., method="eigh"|"svd")`. POD, mPOD, ST-POD and
PSD-POD all call that solver; `lift_kind` metadata comes from `lift.kind`.

**Spatial weights.** Every analyzer accepts `spatial_weight_type` in
`{"uniform", "polar", "prescribed"}` (anything else — including the former
`"auto"` — raises at construction). Omitting the argument (`None`) resolves to
`"uniform"`. Pass an array as `spatial_weights=` to prescribe a metric: the
type becomes `"prescribed"`, and the vector is checked against the snapshot
grid (`n_space`) in `load_and_preprocess` (length/shape, finite, non-negative,
non-zero total). `"prescribed"` without an array, or an array together with
`"uniform"`/`"polar"`, raises at construction. An array with no type still
prescribes. Config/CLI still only expose the type string; prescribing a vector
is a library argument.

**Limitation — uniform W is not a domain integral.** With
`spatial_weight_type="uniform"`, W is the all-ones vector, not cell volumes or
grid spacing. Reported "energy" is therefore a **sum over mesh points**, not a
domain integral, and is **mesh-resolution dependent**: refining the grid changes
the numerical value even when the continuum field is unchanged. Comparing
energies across resolutions requires care (or explicit cell-volume weights via
`spatial_weights=`).

---

## Data Contract

Every analyzer expects a Python dict with these keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `q` | `ndarray (Ns, Nspace)` | yes | snapshot matrix (time × flattened spatial) |
| `x` | `ndarray` | yes | x-coordinates (1D or 2D mesh) |
| `y` | `ndarray` | yes | y-coordinates (1D or 2D mesh) |
| `z` | `ndarray` or `None` | no | z-coordinates for 3D data |
| `dt` | `float` | yes | time step between snapshots |
| `Nx` | `int` | yes | grid points in x |
| `Ny` | `int` | yes | grid points in y |
| `Nz` | `int` | no | grid points in z (default 1) |
| `Ns` | `int` | yes | number of snapshots |
| `t` | `ndarray` | no | time vector |
| `metadata` | `dict` | no | format info, var_name, plot_style, etc. |

### Supported input formats

- **MATLAB `.mat`** — auto-detected via `MATDataLoader`
- **NumPy `.npz`** — consolidated or split layouts via `DNamiDataLoader`
- **Custom loader** — any callable `(file_path: str) -> dict`

### Spatial weights

| Type | When to use |
|------|------------|
| `"uniform"` | Cartesian grids, single-component data (also the default when omitted) |
| `"polar"` | Cylindrical grids (jet nozzle coordinates) |
| `"prescribed"` | Caller-supplied metric via `spatial_weights=` |

`"uniform"` currently returns ones (`calculate_uniform_weights`); it does **not**
apply Δx·Δy cell volumes. Polar weights do use radial spacing. Until cell-volume
weights exist for Cartesian grids, treat POD/SPOD eigenvalues under uniform W as
mesh-dependent sums, not resolution-invariant energies.

---

## Supported Methods

### 1. POD — Proper Orthogonal Decomposition

**Class:** `PODAnalyzer` · **Lift:** identity on centered snapshots · **Operator:** covariance kernel eigenproblem

```python
from openmodalpy import PODAnalyzer

pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
pod.run_analysis()
# pod.modes          — (Nspace, n_modes)
# pod.eigenvalues    — (n_modes,)
# pod.time_coefficients — (Ns, n_modes)
```

**Key facts:**
- Modes are W-orthogonal, ranked by captured weighted variance
- Uses method of snapshots (kernel eigenproblem) when Ns < Nspace
- Mean is subtracted before decomposition

### Non-positive eigenvalues and rank-deficient input

POD, mPOD and PSD-POD share one relative cutoff on the correlation (Gram)
eigenvalues after the eigendecomposition:

\[
\lambda \le n_{\mathrm{kernel}}\,\varepsilon\,\lambda_{\max}
\]

is discarded. Here \(\varepsilon\) is machine epsilon of the working real
dtype (`np.finfo(float).eps` on the default real path), \(\lambda_{\max}\) is
the largest eigenvalue, and \(n_{\mathrm{kernel}}\) is the dimension of the
matrix that was actually factored — `n_samples` on the temporal branch
(`Ns < Nspace`) and on the complex PSD-POD path, `n_space` on the spatial
branch. The same rule replaces the old absolute `1e-12` floor that mPOD used
and the old keep-all behaviour that POD used, so the returned count is scale
invariant (a flow expressed in millimetres yields the same mode count as the
same flow in metres).

Returned modes always have unit weighted norm, and no returned eigenvalue is
negative. When the data supports fewer modes than `n_modes_save` / `n_keep`,
the solver returns the shorter basis — it does not pad with noise directions.
Code that indexes a fixed mode width should use `modes.shape[1]` (or the
length of `eigenvalues`), not assume the request was filled.

**This cutoff is the numerical rank of the correlation matrix, NOT of the
snapshot data.** Measured: a mode at singular-value ratio \(10^{-8}\)
(eigenvalue ratio \(10^{-16}\)) is recovered exactly by the SVD route
(correlation 1.000000, `matrix_rank` 3) while its Gram eigenvalue is
indistinguishable from the noise floor. The snapshot / Gram route cannot
resolve below \(\lambda / \lambda_{\max} \sim n\,\varepsilon\). Do not read the
number of returned eigenvalues as the rank of the snapshot matrix itself.

**Choosing the solver route.** POD defaults to `solver="eigh"`: it factors
the correlation / Gram kernel. Pass `solver="svd"` (or, from a config,
`params: {solver: "svd"}`) to factor the weighted snapshot matrix instead.
ST-POD already uses the SVD route internally and has no user knob.

A singular-value ratio \(r\) is an energy ratio \(r^{2}\). The gap between the
routes therefore starts to matter around energy \(\sim 10^{-14}\) on ordinary
velocity POD (where \(r \sim 10^{-7}\)). Reach for `svd` when that matters:

- delay-embedded / Hankel data (ST-POD’s case; already on this path)
- strongly anisotropic or stretched spatial weights, where the weighted
  condition number is large even when the raw field looks fine
- nearly redundant snapshots from oversampled slow dynamics
- mid-spectrum eigenvector quality, which degrades from squaring before the
  hard \(n\,\varepsilon\) floor is hit

Both routes are \(O(n_{s}^{2}\,n_{x})\). Only the constant differs; `eigh` is
cheaper when \(n_{\mathrm{samples}} \ll n_{\mathrm{space}}\). On well-separated
modes both return the same leading subspace.

```python
pod.perform_pod()                 # default: correlation / Gram (eigh)
pod.perform_pod(solver="svd")     # weighted snapshot SVD
```

### 2. mPOD — Multiscale POD

**Class:** `MPODAnalyzer` · **Lift:** temporal band filtering · **Operator:** POD per band

```python
from openmodalpy import MPODAnalyzer

mpod = MPODAnalyzer(
    file_path="data.mat", n_modes_save=10,
    band_edges=[0.0, 0.15, 0.35, 1.0],  # normalized Nyquist
    band_scale="normalized_nyquist",
)
mpod.run_analysis()
```

**Key facts:**
- Partitions temporal frequency axis into non-overlapping bands
- Current implementation uses rectangular (brick-wall) filters
- Modes live in the same space as POD modes but are scale-separated
- Reference: Mendez et al. (2019), JFM 870

**Limitation — POD-per-band pool, not Mendez MRA.** Each band is POD'd
independently; band modes are concatenated and re-sorted by eigenvalue with no
joint W-orthonormalization. Modes from different bands are not orthonormal
across bands (`Φᵀ W Φ ≠ I` for the pooled set): cross-band inner products are
generally nonzero. Within a single band, POD orthonormality still holds. This is
a simplification relative to the Mendez et al. multiresolution construction; do
not treat the full mode matrix as a W-orthonormal basis.

### 3. PSD-POD — Power-Spectral-Density POD

**Class:** `PSDPODAnalyzer` · **Lift:** pooled blockwise Fourier realizations · **Operator:** single second-order eigenproblem on the flattened Fourier ensemble

```python
from openmodalpy import PSDPODAnalyzer

psd = PSDPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5, n_modes_save=10)
psd.run_analysis()
# psd.eigenvalues        — (n_modes_save,)
# psd.modes              — (Nspace, n_modes_save)
# psd.time_coefficients  — (n_fourier_realizations, n_modes_save)
# psd.freq, psd.St       — frequency and Strouhal axes from the Welch blocks
```

**Key facts:**
- Uses same Welch-block preprocessing as SPOD
- Solves one global eigenproblem instead of per-frequency
- Captures broadband coherent structures
- Triggered via `method="psd-pod"` in config or by constructing `PSDPODAnalyzer` directly

### 4. SPOD — Spectral POD

**Class:** `SPODAnalyzer` · **Lift:** blockwise Fourier transform · **Operator:** per-frequency covariance eigenproblem

```python
from openmodalpy import SPODAnalyzer

spod = SPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
spod.run_analysis()
# spod.eigenvalues — (n_freq, n_blocks)
# spod.modes       — (n_freq, Nspace, n_blocks)
# spod.St          — Strouhal number array
# spod.freq        — frequency array (Hz)
```

**Key facts:**
- Frequency-by-frequency POD of Welch-block spectral ensemble
- Block length controls bias-variance tradeoff of spectral estimator
- Caches FFT blocks in HDF5 for reuse
- Strouhal normalization via `characteristic_length` and `characteristic_velocity` params
- Reference: Towne, Schmidt & Colonius (2018), JFM 847

**Limitation — `dst` is a Strouhal step, not a frequency step.** After the
block FFT, SPOD normalizes by `sqrt(nblocks * dst)` where
`dst = St[1] - St[0] = df · L / U` (not `df = fs / nfft`). Reported eigenvalues
therefore **scale with U/L**. With the default L = U = 1 (and with the shipped
generators), this coincides with a pure frequency-step weight; any other
characteristic scales silently rescale the energy axis.

### 5. ST-POD — Delay-Embedded Space-Time POD

**Class:** `STPODAnalyzer` · **Lift:** delay/Hankel stacking · **Operator:** POD in delay space

```python
from openmodalpy import STPODAnalyzer

stpod = STPODAnalyzer(file_path="data.mat", embedding_dim=10, n_modes_save=10)
stpod.run_analysis()
```

**Key facts:**
- Constructs block-Hankel matrix from centered snapshots
- Metric in lifted space: I_d ⊗ W
- Modes live in C^{d·Nx}; visualization extracts delay-index-0 block
- Uses `compute_reduced_svd` (ARPACK for large matrices)
- Requires uniform dt

### 6. DMD — Dynamic Mode Decomposition

**Class:** `DMDAnalyzer` · **Lift:** identity (shifted pairs) · **Operator:** LS or TLS regression

```python
from openmodalpy import DMDAnalyzer

dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10, rank=10)
dmd.load_and_preprocess()
dmd.perform_dmd(method="ls", delays=1)        # standard DMD
dmd.perform_dmd(method="tls", delays=1)       # TLS-DMD
dmd.perform_dmd(method="ls", delays=4,
                named_variant="hodmd")          # HODMD
dmd.perform_dmd(method="tls", delays=4,
                named_variant="tls_hodmd")      # TLS-HODMD
dmd.save_results()
```

**Rank vs saved modes:**
- `rank` — **required.** SVD truncation of the DMD operator (the reduced system
  size). Pass a positive `int`, `"svht"`, or `"energy"`. Omitting it raises
  `ValueError`. There is no default: the previous silent default of
  `n_modes_save` coupled a plotting parameter to the operator rank, and on the
  shipped cylinder wake that choice moved the recovered shedding frequency by
  ~20×.
- `n_modes_save` — how many modes are kept for save/plot after sorting by `|λ|`.
  Changing `n_modes_save` alone must not change eigenvalues.

| `rank` | Criterion |
|--------|-----------|
| `int` | Explicit rank, floored by the relative cut `s_j > rcond * s[0]` (`rcond = max(shape) * eps`, as in `numpy.linalg.pinv`). Never exceeds what the data supports. |
| `"svht"` | Gavish & Donoho (2014) optimal hard threshold, unknown-noise variant: `τ = ω(β) · median(s)` with `ω(β) = λ(β)/√μ_β`, `μ_β` the Marchenko–Pastur median, and `β = min(shape)/max(shape)`. On data with no coherent signal (flat singular spectrum) this can return `effective_rank == 0`: empty eigenvalues and modes, plus a `RuntimeWarning`, because `τ` then exceeds `σ₁`. That is the honest outcome, not a crash. |
| `"energy"` | Smallest `r` with cumulative `s²` fraction ≥ `energy_fraction` (config key `energy_fraction`, analyzer default `0.999`; `None` in config leaves that default alone). |

**Why there is no default rank:** On the shipped cylinder wake
(`Nx=40, Ny=24, Nt=400`, so `X1` is 960×399) the singular spectrum decays
**smoothly** to `σ_min/σ_1 ≈ 4.5×10⁻⁴` and never approaches the machine floor
(`rcond ≈ 2×10⁻¹³`). Keeping every direction above that floor (rank 399)
produces **spurious modes with `|λ| > 1`** (growth outside the unit circle)
that sort *first* by the amplitude ranking — recovered dominant frequency
**≈ 3.11 Hz** against true shedding **≈ 0.167 Hz**. Truncating at
`rank=10`, `"svht"`, or `"energy"` all recover the physical shedding mode with
`|λ| = 1`. A stability library that manufactures instabilities cannot guess an
untruncated rank. Full rank remains available only by passing an explicit large
`int`.

**Why the library does not default to SVHT either:** SVHT assumes a low-rank
signal plus **i.i.d. Gaussian noise of constant variance**, and its
median-based noise estimate requires the **true rank below n/2** — otherwise
the median singular value is signal, not noise, and the criterion collapses the
rank. Neither assumption holds for a typical deterministic fluid simulation with
a smoothly decaying spectrum. SVHT is also computed from `X1` alone, while the
DMD operator error depends on content of `X2` outside `range(X1)`. So `"svht"`
ships selectable and documented, not as a silent default. No automatic criterion
is free of trade-offs; choose a rank, and report `effective_rank` with the
criterion you used.

**Key facts:**
- Eigenvalues encode frequency (angle) and growth/decay (modulus)
- LS regression assumes noise only in Z+; TLS allows errors on both sides
- Implementation uses broadcasting (`/ s_r`) instead of `np.diag(1/s_r)`
- `named_variant` parameter sets metadata; avoids monkey-patching
- Explicit large `rank` forces the dense SVD path when `min(X1.shape) ≥ 256`
- DMD does **not** subtract the temporal mean and does **not** apply the spatial
  weights `W` in the regression. Both are recorded in the saved metadata
  (`uses_mean_subtraction=False`, `uses_spatial_metric_in_regression=False`).
  On a field with a large steady offset (for example the shipped cylinder wake),
  the mean shows up as a near-unit-circle mode (`|λ| ≈ 1`) with no separate warning.
  If you want a fluctuation-only DMD, center the snapshots yourself before the fit.
- Reference: Schmid (2010), JFM 656; Tu et al. (2014), JCD 1; Hemati et al. (2017), TCFD 31; Gavish & Donoho (2014), IEEE TIT

#### External cross-check

LS and TLS eigenvalues are compared to vendored numbers from **PyDMD 2025.8.1**
(Python 3.12, NumPy 2.5.2, SciPy 1.18.0) on a 12-space, 40-snapshot linear
system built from five chosen eigenvalues. The numbers live in
`tests/fixtures/reference/external_dmd.json`; the comparison is
`tests/test_external_reference.py`. The fixture also stores sha256 fingerprints
of the noiseless and noisy snapshot arrays (float64, C order) so a NumPy
stream change cannot be read as a DMD regression. PyDMD is not a dependency —
the fixture is generated once, outside the repo, by
`scripts/regen_external_reference.py`.
The TLS routes differ algebraically: openmodalpy splits the left singular
vectors of stacked `[X1; X2]`; PyDMD projects both snapshot matrices onto its
leading right singular vectors. Same estimator, different algebra — they agree
to ~1e-15 on noiseless data and only to ~3e-10 under 1e-3 rms noise. That
residual is not a bug.

### 7. HODMD — Higher-Order DMD

Same class as DMD (`DMDAnalyzer`), with `delays >= 2`.

**Key facts:**
- Delay-embeds snapshots into Hankel vectors before forming pairs
- Captures dynamics that appear nonlinear in original coordinates
- Requires uniform dt (non-uniform sampling corrupts the Hankel lift)
- Both LS-HODMD and TLS-HODMD supported
- Reference: Le Clainche & Vega (2017), SIAM J. Appl. Dyn. Syst. 16

### 8. BSMD — Bispectral Mode Decomposition

**Class:** `BSMDAnalyzer` · **Lift:** Hadamard product of Fourier pairs · **Operator:** cross-bispectral eigenproblem

```python
from openmodalpy import BSMDAnalyzer

bsmd = BSMDAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
bsmd.run_analysis()
# bsmd.energy_map   — bispectral energy over (f1, f2)
# bsmd.triads       — identified triadic frequency pairs
# bsmd.eigenvalues  — coupling strength per triad
```

**Key facts:**
- Identifies nonlinear triadic interactions (f1 + f2 = f3)
- Uses dominant eigenpair as practical approximation to numerical-radius problem
- Inspired by [Schmidt's MATLAB BMD](https://github.com/olivertschmidt/bmd)
- Reference: Schmidt (2020), Nonlinear Dynamics 102

**Limitation — default triad list is `ALL_TRIADS` with |p| ≤ 8.** The shipped
static triad table only covers frequency-bin indices with absolute value at most
8. At the default `nfft=128` that is the bottom 12.5% of the rfft spectrum;
higher-frequency triads are not analysed unless you pass a custom
`static_triads` list. Bound: every component must satisfy
`|p| <= min(nfft // 2, n_loaded - 1)` (rfft Nyquist and the bins actually
loaded in `qhat`). With no bins loaded, analysis refuses with a message that
says so rather than quoting a negative bound. The default list
(`static_triads=None`) is **warned and filtered** when any triad falls outside that bound; if filtering leaves none,
analysis raises `ValueError`. A user-supplied list raises `ValueError` naming
every offender. Dynamic triad selection (`use_static_triads=False`) is not
implemented and raises `NotImplementedError`.

---

## Configuration System

### JSONC config structure

```jsonc
{
  // Suite metadata
  "name": "My analysis suite",
  "description": "Optional description",

  // Case definition (shared across all runs)
  "case": {
    "name": "case_name",           // used for output directory names
    "case_type": "experimental",   // or "analytical"
    "data": {
      "kind": "file",              // "file", "generator", or "dnami"
      "path": "../data/file.mat"   // relative to this config file
    },
    "spatial_weight_type": "uniform",  // "uniform" or "polar" (library also accepts "prescribed" + spatial_weights=)
    "n_modes_save": 10,
    "rank": 10,                    // DMD only (required): positive int | "svht" | "energy"
    "energy_fraction": 0.999,      // DMD only: cumulative s² target when rank is "energy" (analyzer default 0.999)
    "nfft": 128,                   // FFT block size (SPOD/BSMD/PSD-POD)
    "overlap": 0.5,                // block overlap fraction
    "embedding_dim": 10,           // delay depth (ST-POD/HODMD)
    "generate_plots": true,
    "results_root": "../results/case_name",
    "figures_root": "../figures/case_name"
  },

  // Runs: each gets its own subdirectory under results/figures root
  "runs": [
    { "id": "pod",      "method": "pod" },
    { "id": "mpod",     "method": "mpod",
      "params": { "band_edges": [0, 0.15, 0.35, 1.0],
                  "band_scale": "normalized_nyquist" } },
    { "id": "dmd_ls",   "method": "dmd",
      "params": { "method": "ls", "delays": 1 } },
    { "id": "dmd_tls",  "method": "dmd",
      "params": { "method": "tls", "delays": 1 } },
    { "id": "hodmd",    "method": "hodmd",
      "params": { "delays": 4 } },
    { "id": "tls_hodmd","method": "tls-hodmd",
      "params": { "delays": 4 } },
    { "id": "spod",     "method": "spod" },
    { "id": "psd_pod",  "method": "psd-pod" },
    { "id": "bsmd",     "method": "bsmd" },
    { "id": "stpod",    "method": "stpod" }
  ]
}
```

### Data source kinds

| Kind | Required fields | Description |
|------|----------------|-------------|
| `"file"` | `path` | Load from `.mat` or `.npz` file |
| `"generator"` | `name`, `params` | Generate at runtime (`double_gyre`, `taylor_green`, `cylinder_wake`) |
| `"dnami"` | `path`, `schema` | Schema-driven dNami NPZ loader (consolidated or split layout) |

### Method aliases

| CLI/config name | Internal ID |
|----------------|-------------|
| `psd-pod` | `psd_pod` |
| `tls-hodmd` | `tls_hodmd` |

All other method names work as-is: `pod`, `mpod`, `dmd`, `hodmd`, `spod`, `bsmd`, `stpod`.

---

## CLI Reference

```
openmodalpy analyze <method> --config <path.jsonc> [options]
openmodalpy run --config <path.jsonc> [--dry-run]
openmodalpy methods list
openmodalpy methods show <name>
openmodalpy examples list
openmodalpy examples show <name>
openmodalpy examples run <name> [--dry-run]
openmodalpy results inspect <path>
```

### analyze options

| Flag | Description |
|------|-------------|
| `--config` | Path to JSONC case config (required) |
| `--run-id` | Custom output subdirectory name |
| `--dry-run` | Preview without executing |
| `--no-plots` | Disable figure generation |
| `--n-modes` | Override n_modes_save |
| `--nfft` | Override FFT block size |
| `--overlap` | Override overlap fraction |
| `--embedding-dim` | Override delay depth |
| `--method ls\|tls` | DMD regression model |
| `--delays` | DMD delay embedding depth |
| `--band-edges` | mPOD band edges (comma-separated) |
| `--band-scale` | mPOD band scale (`hz` or `normalized_nyquist`) |
| `--results-dir` | Override results root |
| `--figures-dir` | Override figures root |
| `--weight-type` | Override spatial weight type |

---

## Implementation Notes

### SVD strategy

`compute_reduced_svd(X, rank)` in `core/base.py`:
- If `rank < min(X.shape)` and `min(X.shape) >= 256`: uses `scipy.sparse.linalg.svds` (ARPACK, truncated)
- Otherwise: uses `np.linalg.svd(X, full_matrices=False)` (dense LAPACK)

Shared by DMD, HODMD, and ST-POD.

### DMD broadcasting

The exact DMD operator `A_tilde = U_r* Z+ V_r Sigma_r^{-1}` is computed as:
```python
atilde = (u_r.conj().T @ X2 @ v_r) / s_r  # broadcasting, not np.diag
modes = X2 @ (v_r / s_r) @ w              # same trick for mode recovery
```

### SPOD FFT caching

SPOD caches blockwise FFT results in the HDF5 output file under the
`FFTBlocks` dataset. Subsequent SPOD runs on the same data with the same
`nfft`/`overlap` skip the FFT computation.

### Delay embedding

`_delay_embed(X, d)` in `dmd.py` builds the Hankel matrix with pre-allocated
output (avoids `np.vstack` of d temporary arrays):
```python
out = np.empty((n * d, cols), dtype=X.dtype)
for i in range(d):
    out[i * n : (i + 1) * n, :] = X[:, i : i + cols]
```

---

## Built-in Generators

| Name | Function | Parameters | Ground truth |
|------|----------|-----------|-------------|
| `double_gyre` | `generate_double_gyre()` | Nx, Ny, Nt, A, epsilon, period, t_max | Known period T, frequency f₀=1/T |
| `taylor_green` | `generate_taylor_green()` | Nx, Ny, Nt, nu, U0, L | Exact decay: λ = e^{-2νΔt}, rank-1 |
| `cylinder_wake` | `generate_cylinder_wake()` | Nx, Ny, Nt, Re, D, U_inf, seed | Known St = 0.212(1 - 21.2/Re) |

### Reproducibility

**What an external user can reproduce from the wheel alone.** The four
example configs that ship in the sdist/wheel are generator-backed and need no
external data files:

- `double_gyre.jsonc`
- `taylor_green.jsonc`
- `cylinder_wake.jsonc`
- `run_benchmarks.jsonc`

**What is not distributed.** These configs live in the source tree for local
development but are excluded from the package (`pyproject.toml` sdist/wheel
exclude list) because they point at benchmark datasets that are not shipped:

- `cavity.jsonc`
- `cylinder.jsonc`
- `cylinder_wake_compressible.jsonc`
- `jet.jsonc`
- `jet_small.jsonc`

An independent check of the three analytic generators (without plots) is the
committed JSON under `tests/fixtures/reference/`. Each fixture stores:

- **POD energy fractions** normalised by the **pre-truncation** total energy
  (recoverable as `sum(kept λ) / energy_captured_fraction`). The fractions
  therefore sum to `energy_captured_fraction` (the share of total energy held
  by the retained modes), not to 1. That makes a leak into modes beyond the
  retained set visible. The full leading-`n_modes` tail is kept even when
  entries sit at ~1e-16 (honest structure on a rank-1 field).
- **DMD** `|λ|` and phase after **canonical ordering**: magnitudes descending;
  eigenvalues whose consecutive `|λ|` agree within the fixture `rtol` form a
  group sorted by phase ascending. The analyzer still emits conjugate pairs in
  LAPACK order; the reference layer reorders only for the golden file so the
  comparison is portable across BLAS/LAPACK emission order.
- `energy_captured_fraction`, `rtol`, and `atol` on the same small fixed grid
  stated in the file.

Shared definition (regen and test): `tests/reference_helpers.py`. Regenerate with:

```bash
uv run python scripts/regen_reference_fixtures.py
```

The comparison test is `tests/test_reference_fixtures.py`; it reads the
tolerance from each fixture (never a literal in the test). Taylor–Green's
recorded `dmd_abs_lambda[0]` is also checked against the generator closed form
`metadata["dmd_eigenvalue"]` (`exp(-2νΔt)`), so that quantity is not golden-only.

**Tolerance.** On a single-thread BLAS policy (the package default), three
in-process re-runs of each generator produced a measured relative spread of
0.0 for POD energy fractions and DMD |λ|, and an absolute phase spread of 0.0.
The fixtures use `rtol=1e-6` and `atol=1e-12`: a large margin above that
measured zero, still far below a 1 % change. Note that `pytest` alone checks
agreement, not sensitivity: confirming the fixtures still *discriminate* means
perturbing a recorded spectrum (1 % on a magnitude, additively on a phase) and
watching the suite go red. Bit-identical results across BLAS
vendors are not promised; see [BLAS thread policy](#blas-thread-policy).

---

## Output Format

Every analyzer writes the same HDF5 dataset names for the same concepts
(lowercase). The single module `openmodalpy.core.results` owns the name table,
the writer, and the generic reader:

```python
from openmodalpy import read_results

res = read_results("path/to/result.hdf5")
# res.modes, res.eigenvalues, res.time_coefficients, res.freq, res.st, ...
# res.attrs["analysis_type"]
```

**Canonical datasets** (present when the method produces them):

| Name | Methods |
|------|---------|
| `modes`, `eigenvalues`, `time_coefficients` | POD, mPOD, ST-POD, DMD, SPOD, PSD-POD |
| `freq`, `st` | SPOD, PSD-POD |
| `amplitudes`, `omega` | DMD |
| `modes1`, `modes2`, `triads` | BSMD |
| `x`, `y`, `z`, `W`, `temporal_mean`, `energy_map` | when available (already uniform) |
| `FFTBlocks` | SPOD/BSMD FFT cache (name unchanged on purpose) |

**Attributes** (unchanged): `analysis_type`, `nfft`, `overlap`, `dt`, `Ns`,
`Nx`, `Ny`, `spatial_weight_type`, method-specific metadata.

DMD also records `dmd_variant`, `dmd_method`, `dmd_delays`, `dmd_named_variant`.

**Provenance** — every file written through `write_results` also carries a
`prov_*` block describing the software that produced it. Read it as
`read_results(path).provenance` (prefix stripped). Files written before this
exist report an empty mapping; missing keys never raise.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `prov_openmodalpy_version` | str | Running package `__version__` (metadata fallback) |
| `prov_python_version` | str | Running CPython X.Y.Z |
| `prov_numpy_version` | str | Installed NumPy version |
| `prov_scipy_version` | str | Installed SciPy version |
| `prov_h5py_version` | str | Installed h5py version |
| `prov_fftkit_version` | str | Installed fftkit version |
| `prov_fft_backend` | str | `fftkit.DEFAULT_BACKEND` at write time |
| `prov_blas_threads` | int | Effective BLAS thread limit used by kernels (`0` = no package limit / observation failed) |
| `prov_config_sha256` | str | SHA-256 of analysis attrs (not data); excludes `prov_*` |
| `prov_created_utc` | str | UTC write timestamp (`YYYY-MM-DDTHH:MM:SSZ`) |
| `prov_git_sha` | str | openmodalpy package checkout HEAD when available, else `unavailable` |
| `prov_seed` | str | Analysis seed when present (`data_seed`/`seed`), else `none` |

`save_results(self, filename=None)` is the uniform writer signature on every
analyzer. Files written with the older capitalised names (`Modes`,
`Eigenvalues`, `TimeCoefficients`, `Freq`, `St`, `Modes1`, `Modes2`, `Weights`)
still load through `read_results`, which maps them onto the canonical fields
and emits a `DeprecationWarning`.

---

## Testing

Key test categories:

| File | What it tests |
|------|--------------|
| `test_pod.py` | POD eigenvalues, mode shapes, energy convergence |
| `test_mpod.py` | Band filtering, frequency separation |
| `test_dmd.py` | Exact DMD, TLS, delays, HODMD metadata, roundtrip |
| `test_stpod.py` | Delay embedding, Hankel shape, validation |
| `test_spod_plot.py` | SPOD plotting paths |
| `test_bsmd_core.py` | BSMD triad detection, energy map |
| `test_cli_commands.py` | CLI dispatch, config parsing, dry-run, PSD-POD metadata |
| `test_provenance.py` | Provenance block on all five analyzers; hash/prov-independence; never-raise; backend; unknown threads=0; legacy empty view |
| `test_dnami_loader.py` | NPZ loading, schema handling |
| `test_weights.py` | Polar and uniform weight computation |
| `test_reference_fixtures.py` | POD/DMD spectra vs committed analytic fixtures |
| `test_external_reference.py` | Vendored PyDMD eigenvalues (LS/TLS, noiseless/noisy) |

Run all: `uv run pytest tests/ -q`

To count them: `uv run pytest -q --collect-only`

---

## Dependencies

**Runtime:** `numpy`, `scipy`, `matplotlib`, `h5py`, `tqdm`

**Dev:** `pytest`, `ruff`

**No external solvers.** All linear algebra uses NumPy/SciPy. Future versions
may optionally use SLEPc (`slepc4py`) for distributed eigensolvers.

---

## Extension Paths

To add a new decomposition:

1. **Variance-optimal method** — write a new `Lift` in
   `core/decomposition.py` (or a thin wrapper), then call
   `weighted_second_order` with the appropriate metric and method.
2. **Evolution-fit method** — write a new lift for paired data, then reuse the
   SVD-based regression in `dmd.py`.
3. **Interaction method** — write a new lift for higher-order objects, then
   implement the coupling optimization.

In all cases:
- Subclass `BaseAnalyzer`
- Register in `METHOD_REGISTRY` in `commands.py`
- Add a `_run_*` function and a dispatch entry in `analyze_from_spec`
- Add a JSONC example config
- Add tests
