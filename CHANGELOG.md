# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- `perform_dmd` now takes `embedding_dim` in place of `delays`. The old keyword raises TypeError.
- Config run params now use `embedding_dim` in place of `delays`.
- Saved DMD results now store `dmd_embedding_dim` in place of `dmd_delays`.
- CLI flag `--delays` is removed. Use `--embedding-dim`.

### Removed

Six functions that nothing called. Each was checked for `__all__`, package
exports, string dispatch and the docs before it went.

- `load_dnami_data`. The two loaders it mirrors, `load_jetles_data` and
  `load_mat_data`, are re-exported from `core.base` and stay. This one never
  was, so nothing can reach it.
- `compute_aspect_ratio` and `get_aspect_ratio`. A dead pair: the only caller
  of the first was the second. `get_fig_aspect_ratio` is the live helper.
- `parallel_map`.
- `_calculate_weights_openmp`, a placeholder that only called
  `_calculate_weights_numpy`.
- `blocksfft_optimized`. `base.blocksfft` is now the one public name for a
  windowed block FFT. `core/welch.py` used to say both names stay public; it
  no longer does.

### Added

- A speed and memory tripwire test runs three cases and measures each one:
  DMD with a delay embedding, POD, and SPOD. Each case has a wall-time
  ceiling and a peak-memory ceiling taken from measured numbers, with a wide
  margin, so the test only fails when a change makes a case several times
  slower or several times larger. The measured numbers print after the run
  on a `PERF` line, so a slow machine is easy to tell from a slow change.
- DOC.md says that `solver="eigh"` and `solver="svd"` are two different
  numerical routes, and when to prefer each.
- DOC.md and README say that `hodmd` and `tls-hodmd` are `DMDAnalyzer`
  settings, and show the call. DOC.md adds a table of the array shapes
  each method returns.
- SPOD closed-form check now runs the default 0.5 block overlap, and the two
  other overlaps around it.
- `SPODAnalyzer` accepts `n_modes_save`. SPOD makes one mode per Welch block at
  each frequency, and `modes` is the largest array it writes. `n_modes_save`
  keeps the leading modes and cuts `modes` and `time_coefficients` on their last
  axis. Eigenvalues keep every block, because the spectrum figure draws one line
  per block. The default keeps every mode, so results do not change unless you
  set it. A value above the block count keeps every block and reports a
  `RuntimeWarning` naming both numbers.
- `examples/compare_pod_spod.py` is the worked example for the package's
  core promise: build one dataset from the shipped double-gyre generator,
  hand it to `PODAnalyzer` and `SPODAnalyzer` through `data=`, and plot
  both. A test runs it, so it cannot go stale.
- `examples/my_data_template.py` is a template for your own data format. It
  names every contract key, says which are required, and states how `q` must be
  flattened. A test runs it, so it cannot go stale. DOC.md, "Your own format",
  explains the three steps.
- DOC.md now states one plug-in point for your own reader: a plain callable
  `(path) -> dict`, given as `data_loader=` or called yourself and passed as
  `data=`. The `DataLoader` base class and `DataInterfaceManager` are named as
  internal, which is how the shipped readers are written.
- `generate_double_gyre`, `generate_taylor_green`, `generate_cylinder_wake`,
  `generate_example_dataset`, `get_example_info`, and `load_example_payload`
  are now exported from the top-level `openmodalpy` package.
- The provenance block written to every result file now also records which
  BLAS/LAPACK library actually ran (`prov_blas`), the OS platform and machine
  architecture (`prov_platform`, `prov_machine`), and the HDF5 C library
  version (`prov_hdf5_version`).
- `scripts/mutation.sh` runs mutation testing over the numerical core
  (`core/decomposition.py` and `core/welch.py`). It reports how many mutants
  the test suite killed and how many survived. Coverage says which lines the
  tests run; this says whether the tests fail when the numbers are wrong. The
  run takes approximately an hour, so it stays off the per-push path: the
  `mutation` workflow runs it on demand and on the first day of each month.
  The first recorded run killed 699 of 812 mutants (86%); 109 survived and 4
  timed out. The survivors group in the solver routing and the tolerance
  helpers. DOC.md, "Mutation testing", holds the baseline table to compare
  later runs against.

### Changed

- `load_results` now reads the result file once. Every method rebuilt the
  file name after the base class had already read the file, then read it a
  second time to reach a few fields. A POD load opened the file twice and
  built every array twice; so did SPOD, ST-POD, DMD, PSD-POD and BSMD. The
  fields each method wants now come from the first read. Six `load_results`
  overrides are gone, and each method names the datasets it demands instead
  of writing its own check.
- BSMD checks the conjugation stamp before it assigns anything, so a file
  from a pre-fix build no longer leaves the analyzer half filled.
- ARPACK now gets a contiguous matrix. It does one matrix-vector product per
  iteration, and a sliced view made every one of them stride through memory.
  DMD with `embedding_dim=4` on a 64 MB field fell from 2.559 s to 0.557 s.
- Every transform now goes through fftkit, and no module calls numpy FFT directly.
  Welch asks for the one-sided transform instead of computing the full complex one
  and discarding half of it. The results do not change.
- DMD, HODMD and ST-POD now share the name `embedding_dim` for delay embedding depth. DMD default is 1 and does no lift. ST-POD still rejects 1.
- The SPOD eigenproblem is now one hop from `spod.py`. `spod.py` imports
  `spod_single_frequency` from `openmodalpy.core.decomposition` and calls it
  directly. It used to call `spod_function` in `openmodalpy.core.base`, a file
  that holds no SPOD mathematics, which then called one of two entry points
  that ran the same body. `spod.py` also gains a module docstring that states
  the eigenproblem and says which module holds it.
- Result files written by `MPODAnalyzer`, `DMDAnalyzer` and `STPODAnalyzer`
  now record `nfft=1` and `overlap=0.0`, in place of the former default
  `nfft=128` and `overlap=0.5`. These methods never form an FFT block, so the
  old numbers named a block size that was never used. The value is a stamp for
  the shared filename and metadata helpers, not a setting.
- `SPODAnalyzer.perform_spod()`, `PSDPODAnalyzer.perform_psd_pod()`, and
  `BSMDAnalyzer.perform_bsmd()` now form their own FFT blocks on first use.
  Calling `compute_fft_blocks()` before them is no longer needed: right after
  `load_and_preprocess()`, the `perform_*` call works on its own.
- CI: mypy now runs on macOS and Windows as well as Linux; a weekly
  unpinned-resolution run is informational only and does not gate merges.
- The sqrt(W) weighting of a samples x features matrix now lives in one
  place (`apply_sqrt_metric`, `core/decomposition.py`); ST-POD's total
  energy and the SVD solver both call it instead of each re-deriving it.
- `spatial_weight_type="uniform"` is back to all-ones weights (v0.5.0
  behaviour). The trapezoid cell-volume metric is now the explicit opt-in
  type `"cell_volume"`; without usable 1-D grid coordinates it raises.
- `spatial_weight_type="polar"` now accepts scattered points (1-D `x`, `y`
  of length `Nspace`) and weights each point by its radius, `w_i = |y_i|`;
  grid input is unchanged.
- `spatial_weight_type="polar"` on a 3-D grid now reads `z` as azimuth theta
  in radians and weights each (x, r, theta) cell as a sector of the (x, r)
  annulus; a `z` range beyond one revolution (2*pi) raises.
- The generic loader now infers scattered points from 1-D `x`/`y` of length
  `Nspace` whose product does not match `Nspace`, without needing `Nx`/`Ny`
  stated in the file.
- The test-suite coverage floor moved from a nominal 50% (24 points under the
  measured value, so real regressions stayed invisible) to a 72% ratchet read
  from `pyproject.toml` by both local runs and CI. The ratchet policy — when
  the floor moves and who moves it — is written down in CONTRIBUTING.
- The library entry point and the command line now execute ONE analysis
  sequence per method (`run_analysis`), so the two paths can no longer drift
  apart in plotting or save behaviour.
- POD and ST-POD now read a square spatial weight matrix through the same
  shared rule as everywhere else: only an exactly diagonal square is
  accepted (its diagonal is used); any other square raises. The loose
  `np.allclose` that used to wave through a small off-diagonal is gone.

### Fixed

- DOC.md pointed to `commands.py` for `METHOD_REGISTRY`. It lives in `specs.py`.

- `prov_blas` used to report a thread count too, read from the idle pool at
  collection time. That count could contradict `prov_blas_threads`, the
  actual cap applied. `prov_blas` now names only the BLAS or LAPACK build
  that ran.
- The documentation called TLS-DMD "de-biased for noisy data" and set no bound
  on it. That advantage holds at `embedding_dim=1` and decays as `embedding_dim` grows,
  because delay embedding repeats the same noise across the Hankel rows while
  TLS assumes the two snapshot matrices carry independent errors. Measured over
  200 seeds, TLS beat LS in 177/200 runs at `embedding_dim=1` and in 95/200 at
  `embedding_dim=5`, where LS is better on average. README.md, DOC.md and the
  `perform_dmd` docstring now say so. No solver changed.
- The `BSMDAnalyzer` class docstring said BSMD "keeps one mode per triad, so
  the mode count follows from the triad count, not a chosen number". Keeping one
  mode per triad is the approximation this analyzer makes, not a property of the
  operator: the assembled matrix is `(n_blocks, n_blocks)` and has `n_blocks`
  eigenpairs, of which the dominant one is selected. The triad count is also a
  chosen number, because the caller passes `static_triads`. The docstring now
  says both, and a test pins that the result count follows the triad list and
  not the block count.
- `SPODAnalyzer.perform_spod` said the computation is "delegated to the
  `spod_function` imported from `utils.py`". There is no `utils.py` in this
  package, so a reader following that sentence reached nothing.
- The `SPODAnalyzer` class docstring gave the wrong axis order for
  `time_coefficients`. It said `(n_freq, n_modes, n_blocks)`; the array is
  `(n_freq, n_blocks, n_modes)`, because the per-frequency eigenproblem is
  solved in block space. Both axes have the same length when nothing is
  truncated, so the error was invisible until `n_modes_save` made them differ.
- The `SPODAnalyzer` class docstring said SPOD takes no `n_modes_save` and that
  its mode count cannot be chosen. It now describes the option and states that
  the Welch block count is the ceiling.
- A custom `data_loader=` callable did not get the derived counts, so a loader
  that returned only `q`, `x`, `y` and `dt` still failed with a bare
  `KeyError: 'Ns'`. DOC.md documents `data_loader=` and `data=` as the same
  plug-in point, so both now fill the counts through one rule. A dict that
  already states every count is left alone.
- An analyzer given an already-loaded dict through `data=` raised a bare
  `KeyError: 'Ns'` deep inside the Welch block setup. The same dict read from a
  file worked, because the file reader derives `Nx`, `Ny`, `Nz` and `Ns` from
  the array shapes. Both paths now use one derivation rule, so a dict built by
  hand behaves like a dict read from disk. Only `q`, `x`, `y` and `dt` are
  required. A dict that misses `q`, `x` or `y` raises `ValueError` and names
  the missing keys. A dict that misses `dt` raises when the analyzer loads,
  with the message that names the data source and asks for a positive finite
  scalar. The example printed in README.md ran into this and now runs as
  printed.
- The `prov_blas` format test failed on macOS. `_blas_identity` returns the
  sentinel "unknown" when threadpoolctl reports no bound threadpool, which is
  what a macOS wheel linked against Accelerate does, but the test did not
  accept that value. The test now accepts the sentinel as the whole text, and
  checks the record shape of every other entry instead of only one of them, so
  a malformed entry can no longer hide behind a well-formed one.
- `run_analysis` with `plots=False` no longer reports that plotting completed.
- **Numerical change: polar grid weights.** `spatial_weight_type="polar"`
  on a grid flattened the 2-D (x, r) weights x-major (`index = ix*Ny + iy`)
  while the snapshot contract is y-major (`index = iy*Nx + ix`), so every
  column of `q` was paired with the weight of the transposed cell. Affected:
  every grid result computed with `spatial_weight_type="polar"` (POD, SPOD,
  mPOD, ST-POD, PSD-POD eigenvalues, energies and modes), square grids
  included; scattered points and `"uniform"`/`"prescribed"` are not. Measured
  size: on a 3 x 4 grid the energy of a smooth field was under-weighted by
  12 %; on a 5 x 5 grid single weights differed by up to a factor 3.7; the
  total measure (sum of all weights) was unchanged. To see whether your
  result moved, compare the old and new weight vectors for your grid:
  `W_new = calculate_polar_weights(x, r, use_parallel=False)` (import it with
  `from openmodalpy.core.base import calculate_polar_weights`) against
  `np.reshape(np.outer(Wx, Wy), (-1, 1))` built the old way — any
  difference beyond round-off means your polar eigenvalues change with
  this release.
- BSMD triad plots picked their order with a plain `argsort` on the
  per-triad dominant eigenvalue magnitude, so triads tied to machine
  precision (a common case: eigenvalues come from independent `eig` calls
  that can round differently on different platforms) could swap places
  near the plot cutoff and change which modes made the figure. Tied triads
  now sort by their `(p, q, r)` triad tuple, giving the same figure on
  every platform.
- `BSMDAnalyzer` did not check `overlap`. `overlap=10` silently set
  `novlap=80` against `nfft=8`, an overlap bigger than the block itself.
  It now raises `ValueError: Overlap must be between 0 (inclusive) and 1
  (exclusive).`, the same check and message `SPODAnalyzer` already used.

### Breaking

- `openmodalpy.core.base.spod_function` and
  `openmodalpy.core.parallel.spod_single_frequency_optimized` are removed. Call
  `openmodalpy.core.decomposition.spod_single_frequency` instead; the argument
  order `(qhat, nblocks, dst, w)` and the results are unchanged, so a call that
  passed `use_parallel=True` (the default) only drops that argument.
  `spod_function` chose between two routes that reached the same body, and
  `spod_single_frequency_optimized` forwarded its arguments and did nothing
  else, so neither route ever gave a different answer. Measured: SPOD
  eigenvalues, modes, time coefficients and frequencies on the shipped
  cylinder-wake case are byte-identical before and after.
- `nfft` and `overlap` are no longer accepted by `PODAnalyzer`,
  `MPODAnalyzer`, `DMDAnalyzer`, or `STPODAnalyzer`. They never used these
  Welch block-size settings; passing either now raises `TypeError`. Only
  `SPODAnalyzer`, `PSDPODAnalyzer`, and `BSMDAnalyzer` form FFT blocks, and
  keep `nfft`/`overlap` as their own constructor keywords.
- `use_parallel` is no longer accepted by `PODAnalyzer`,
  `MPODAnalyzer`, `DMDAnalyzer`, `STPODAnalyzer`, `SPODAnalyzer`, or
  `PSDPODAnalyzer`; it never changed their result. `BSMDAnalyzer` keeps
  `use_parallel`, since it really runs its triad loop in a thread pool.
- `DMDAnalyzer.run_analysis()` and `BSMDAnalyzer.run_analysis()` now produce
  figures by default; their docstrings previously promised no default plots.
- `PSDPODAnalyzer.run_analysis()` now produces its standard figure set
  (eigenvalues, cumulative energy, mode slices or 3-D views).
- POD's default library figure set is the one the CLI always produced
  (eigenvalues, volumetric-or-mode panels, time coefficients, cumulative
  energy). The richer library-only defaults (reconstruction grid, pair-phase,
  99.5% modes grid) remain available as explicit `plot_*` calls but are no
  longer part of `run_analysis`.
- `run_analysis(**kwargs)` forwards keyword arguments to the decomposition
  call; the old per-plot keyword parameters (`plot_n_modes_spatial`,
  `plot_modes_options`, `check_orthogonality`, ...) are gone. Pass plotting
  choices through the plot methods themselves.
- Every analyzer constructor now takes only `file_path` positionally; every
  other parameter is keyword-only. The old positional slots meant different
  things per class — `Analyzer(path, 256, 0.5)` bound to `nfft, overlap` on
  `SPODAnalyzer` and `BSMDAnalyzer`, but to `results_dir, figures_dir` on
  `PSDPODAnalyzer`, and a call like `Analyzer(path, 8, 10)` bound to
  `embedding_dim, n_modes_save` on `STPODAnalyzer` but to `nfft, overlap` on
  `BSMDAnalyzer`, with no error either way. A positional call past
  `file_path` now raises `TypeError` immediately instead of silently
  binding to the wrong parameter. `n_modes_save` stays absent from
  `SPODAnalyzer` and `BSMDAnalyzer`: their mode count comes from the block
  count and the triad count, not a chosen number.

  Each row translates the same call. Read the old slot order from your own
  code: the second positional argument was NOT the same parameter in every
  class, which is the reason for this change.

  | Class | Old positional call | New keyword call |
  | --- | --- | --- |
  | POD | `PODAnalyzer(path, rdir, fdir, loader, wtype, 10)` | `PODAnalyzer(path, results_dir=rdir, figures_dir=fdir, data_loader=loader, spatial_weight_type=wtype, n_modes_save=10)` |
  | mPOD | `MPODAnalyzer(path, rdir, fdir, loader, wtype, 10)` | `MPODAnalyzer(path, results_dir=rdir, figures_dir=fdir, data_loader=loader, spatial_weight_type=wtype, n_modes_save=10)` |
  | DMD | `DMDAnalyzer(path, rdir, fdir, loader, wtype, 10, 4)` | `DMDAnalyzer(path, results_dir=rdir, figures_dir=fdir, data_loader=loader, spatial_weight_type=wtype, n_modes_save=10, rank=4)` |
  | ST-POD | `STPODAnalyzer(path, 8, 10)` | `STPODAnalyzer(path, embedding_dim=8, n_modes_save=10)` |
  | SPOD | `SPODAnalyzer(path, 256, 0.5)` | `SPODAnalyzer(path, nfft=256, overlap=0.5)` |
  | BSMD | `BSMDAnalyzer(path, 256, 0.5)` | `BSMDAnalyzer(path, nfft=256, overlap=0.5)` |
  | PSD-POD | `PSDPODAnalyzer(path, rdir, fdir, loader, wtype, 256, 0.5)` | `PSDPODAnalyzer(path, results_dir=rdir, figures_dir=fdir, data_loader=loader, spatial_weight_type=wtype, nfft=256, overlap=0.5)` |

  Note the PSD-POD row. `PSDPODAnalyzer(path, 256, 0.5)` did NOT set `nfft`
  and `overlap`: slots 2 and 3 were `results_dir` and `figures_dir`, so that
  call set a directory named "256". If you wrote it, you had a defect. Give
  `nfft` and `overlap` by keyword, and check which directories you meant.

## [0.5.0] - 2026-08-18

### Added

- HODMD and TLS-HODMD are now checked against systems whose eigenvalues are
  known ahead of time, so "how do you know this is right" has a test behind it
  rather than a claim. One of the cases observes a two-oscillator system
  through a single point, where the delay embedding is what makes the spectrum
  recoverable at all and plain DMD cannot reach it.
- The total-least-squares variant is checked to be the more accurate of the two
  on noisy data, compared as a median over many noise draws rather than a
  single lucky one. Note this holds without delay embedding; stacking delays
  correlates the noise and erodes the advantage.
- TLS-HODMD is also pinned as a genuinely different computation from the plain
  least-squares path when delays are stacked, so the two cannot quietly become
  the same code without a test noticing.
- SPOD eigenvalue magnitudes — the modal energies you read off a spectrum — are
  now checked against a field built to have a known answer, so the size of an
  energy is evidence rather than a claim. Two modes share a frequency with
  different prescribed energies, a second frequency carries a third, and the
  check runs under three window shapes and both window normalisations.
- The documented behaviour that SPOD energies scale with velocity over length,
  because the spectral step is a Strouhal step, is now a checked property
  instead of a warning in the documentation.
- DMD eigenvalues, least squares and total least squares, are now compared
  against numbers from PyDMD instead of only against this package's own
  reasoning. The reference values are produced once outside the repository and
  committed as readable text beside the versions, seeds and solver options that
  made them, so the package gains no dependency and anyone can regenerate them
  and get the same bytes. The noisy case is the one that earns the effort: there
  is no closed form for the estimate, so PyDMD's answer is independent evidence,
  and the tolerance is placed inside the gap between the two estimators, which
  means a total-least-squares path that quietly became least squares fails the
  check. The reference fields ship with the numbers in the same file, so a
  change in how the field is built reports itself rather than looking like a
  DMD error.
- SPOD eigenvalues are compared against PySPOD in the same way, and the
  documentation now states the conversion between the two packages instead of
  leaving you to find it: our eigenvalue is PySPOD's times the number of DFT
  points times the timestep, divided by two. The first factor is our division
  by the Strouhal step, which PySPOD does not do; the two is PySPOD's doubling
  of interior bins, which we do not do. One difference cannot be removed: the
  two packages use different Hamming windows, PySPOD the symmetric definition
  and this package the periodic one, so after the conversion the two still
  differ by about a part in a thousand. That number is written down with the
  comparison rather than hidden in a tolerance, and the documentation says
  which of the two checks is the strict one.

### Changed

- A square array of spatial weights that is **not** diagonal now raises instead
  of being quietly reduced to its diagonal. The rule has not changed — a square
  weight array has always meant its diagonal — but until now the package
  enforced that by discarding your off-diagonal entries without saying so. If
  you were relying on that, pass ``np.diag(W)`` and nothing else changes; if you
  meant a genuinely coupled inner product, the package cannot represent one and
  now tells you rather than silently answering a different question. This
  applies at every door: the ``spatial_weights`` argument, loading a results
  file, and the lower-level decomposition entry points. Note that a uniform
  metric written as ``np.ones((n, n))`` is caught by this and should be
  ``np.ones(n)``.
- The same check covers a 3-D stack of weight planes, where each plane must be
  diagonal on its own.
- Whether a matrix counts as diagonal is now measured relative to the entries
  being coupled, rather than against the largest weight anywhere in the array.
  A metric on a strongly graded mesh no longer hides real coupling between its
  small cells behind one large cell elsewhere, and rescaling a metric cannot
  change the verdict.
- Weights are judged at the precision they are stored in, so a single-precision
  metric is no longer rejected for round-off that is invisible at its own
  precision.
- A weight matrix containing ``NaN`` or infinity off the diagonal now raises.
  Previously these passed through and the diagonal was used regardless.

### Fixed

- DMD mode order no longer depends on the LAPACK build. Conjugate pairs and
  other ``|λ|`` ties are ordered by ``(Re, Im)`` after magnitude, so the same
  data produces the same column order on every machine.
- mPOD mode order no longer depends on which of two nearly-equal band
  eigenvalues wins a rounding comparison. Tied modes are ordered by band
  index, then by their position inside the band, so the same data produces
  the same column order on every machine.

## [0.4.0] - 2026-08-13

One theme: the library used to guess, substitute or quietly repair bad input and
then report success. It now refuses. A fabricated timestep, an ignored window
function, a discarded spatial metric, a silently defaulted DMD rank and a
mistyped config key each produced a confident number that nobody had asked for.
Those paths now raise, and the reasons are named.

Read the upgrade section below first: 16 call sites raise until you change them,
and 9 behaviours move published numbers with no code change at all.

### Upgrading from 0.3.0

Rename or edit — these raise until you change the call:

| 0.3.0 | 0.4.0 |
| --- | --- |
| `SPODAnalyzer.plot_eigenvalues_v2(...)` | `SPODAnalyzer.plot_eigenvalues(...)` |
| `SPODAnalyzer.plot_time_coeffs(...)` | `SPODAnalyzer.plot_time_coefficients(...)` |
| `STPODAnalyzer.plot_time_coefficients(n_coeffs=...)` | `...(n_coeffs_to_plot=...)` |
| `BSMDAnalyzer.save_results(fname=...)` | `save_results(filename=...)` |
| `blocksfft(..., use_parallel=...)` | `blocksfft(...)` — argument removed |
| `weighted_second_order(..., drop_nonpositive=...)` | use `n_keep` to bound the modes |
| `BaseAnalyzer/SPODAnalyzer/blocksfft(n_threads=...)` | `set_blas_threads(...)` |
| `spatial_weight_type="auto"` | omit it, or `"uniform"` |
| `auto_detect_weight_type(...)` | deleted — no honest detection exists |
| `DMDAnalyzer(...)` with no `rank` | `rank=<int>` \| `"svht"` \| `"energy"` — now required |
| `get_optimization_info()`, `print_optimization_status()` | `get_threadpool_summary()` |
| `python -m openmodalpy.pod` (and `.spod`, `.dmd`, `.bsmd`, `.stpod`) | `python -m openmodalpy` |
| `from openmodalpy.core.config import FFT_BACKEND` | `from openmodalpy.core import FFT_BACKEND` |
| `SpatialMetric(<square matrix>)` | `SpatialMetric(np.diag(W))` |
| `loader.load(path, <positional option>)` | options are keyword-only |
| `perform_spod()` before `compute_fft_blocks()` | call `compute_fft_blocks()` first |

`rank=n_modes_save` reproduces the old DMD default exactly.

Numbers that move without any code change:

- **Welch block counts.** `nblocks` now floors and drops the remainder, matching
  `scipy.signal.welch`, instead of ceiling with a clamped final block that reused
  samples. The shipped `cylinder_wake` (`Ns=500`, `nfft=128`, `overlap=0.5`) goes
  from 7 blocks to 6, so SPOD, PSD-POD and BSMD results on records that do not
  divide evenly all shift.
- **POD and ST-POD energy percentages** are now a share of the total field energy
  before truncation, so they read lower for any truncated spectrum.
- **Shipped `double_gyre` DMD rank** moved from 10 to 8. At rank 10 the retained
  singular values reach `s9/s0 ~ 2e-12`, so round-off is amplified to ~1e-4 —
  a hundred times looser than the fixture tolerance. `dmd`, `hodmd` and
  `tls_hodmd` now return 8 modes. Other methods read `n_modes_save`, still 10.
- **Zero-measure cells contribute nothing.** The absolute `1e-12` weight floor is
  gone. A masked cell holding `1e6` used to shift the eigenvalues by 199 %.
- **Rank-deficient input returns fewer modes**, and energy fractions shift by
  about `1e-16` relative, because the noise tail is no longer counted as modes.
- **Mode signs and phases are canonical**, so a mode shape can no longer flip
  between LAPACK builds or thread counts.
- **mPOD figures are named `_mpod_`**, not `_pod_`.
- **The library writes nothing to stdout.** Every analyzer reports on its module
  logger. The command-line tool installs a handler, so its output is unchanged;
  a library caller now routes, filters or silences messages like any other
  Python logging.
- **Old result files still load** but emit a `DeprecationWarning` naming the
  legacy key. Code that turns warnings into errors needs a filter.

### Breaking

- Config loading rejects a key that nothing reads. Unknown keys at the top level,
  in `case`, in `case.data` and in each run entry raise `ValueError` naming the
  file, the key and the keys accepted at that level, so `n_modes_sav` no longer
  falls through to the default. A `spatial_weights` array in a config raises and
  points at the library API. Top-level `kind` is now read: it must be
  `"analysis-suite"` or `"config-suite"` and must agree with whether the file
  carries a `configs` list.
- Loading stops when the spatial metric does not carry exactly one weight per
  column of the snapshot matrix. That metric enters the inner product and its
  length was never checked against the data. Runs that used to complete and now
  stop: a 3-D field whose `z` array is absent while `Nz > 1`; a leftover `z`
  beside a 2-D snapshot matrix; any custom loader whose coordinates disagree with
  the grid keys it reports. Only POD and ST-POD got that far, and they replaced
  the wrong-length metric with unit weights inside the solver — right by accident,
  while the metric on the analyzer was wrong. Runs that already failed now fail at
  load with a clear message: 3-D polar (the polar builder ignores `z`), and any
  SPOD or BSMD run.
- `spatial_weight_type="auto"` is removed. Detection cannot be done honestly from
  the loader contract — a jet `(x,r)` grid and a flat-plate `(x,y)` grid both have
  `y >= 0`, and no coordinate-system metadata is carried — so the former default
  resolved to `"uniform"` unconditionally. The default is now `None`, which still
  resolves to `"uniform"`; numerics are unchanged. An array with no type
  (`spatial_weights=w`) still prescribes a metric.
- An unrecognised `spatial_weight_type` raises at construction. Any other string
  used to fall through to the grid-spacing path and skip POD's and ST-POD's reset
  to unit weights, so a typo silently changed the metric. Pass `spatial_weights=`
  to hand an analyzer its own vector.
- Spatial weights are used exactly, with no absolute `1e-12` floor. The weights
  are a quadrature measure (m³ in 3-D, m² in 2-D), so an absolute floor carries
  units it cannot have: a mesh whose cells fall below `1e-12` was silently
  inflated, and scaling a metric by `1e-14` changed the eigenvalues by a factor
  of about 95. A zero-measure cell now contributes nothing and its POD/ST-POD mode
  value is exactly `0`, matching SPOD and BSMD.
- An invalid spatial metric raises instead of producing a confident answer. A
  non-finite weight, a negative weight, or a metric whose total measure is zero
  raises `ValueError`; an isolated zero among positive weights is still accepted.
  This closes a real hole: polar weights on a grid whose radial coordinate is zero
  give every annulus an area of `pi*r**2 = 0`, and POD reported an energy fraction
  off that empty metric without complaint. The condition is `r > 0`, **not**
  `Ny > 1` — a single radial station at `r > 0` is fine. Results for strictly
  positive weights are unchanged.
- `SpatialMetric` rejects square matrices and 3-D weight arrays with a named
  `ValueError` carrying the shape and the `np.diag` fix, instead of flattening
  them and failing later on length. 1-D, `(n, 1)` and non-square `(n, k)` inputs
  are unchanged.
- One HDF5 result contract for every analyzer. Dataset names are lowercase
  (`modes`, `eigenvalues`, `time_coefficients`, `freq`, `st`, `modes1`, `modes2`);
  SPOD writes `W` rather than `Weights`. Every `save_results` shares the signature
  `save_results(self, filename=None)`. `read_results(path)` returns a typed
  `AnalysisResults` and still accepts the old capitalised layout with a
  `DeprecationWarning`. `FFTBlocks` keeps its name — it is a cache key, not a
  result field. SPOD result files are written in mode `"w"`; BSMD still appends
  when the destination is the open FFT-cache path, so that cache survives.
- SPOD result files write the spatial grid once, as `x`/`y`/`z`. The duplicate
  `x_coords`/`y_coords`/`z_coords` datasets are gone. Files carrying only the old
  spelling still load, mapped onto the canonical fields with a
  `DeprecationWarning`. Files written by earlier versions carry both spellings, so
  reading one now warns where it did not before; the canonical names win, so
  values are unchanged.
- `DataLoader.load` / `MATDataLoader.load` / `DNamiDataLoader.load` options
  (`preview_ns`, `field`, `load_single`, `schema`) are keyword-only. A positional
  second argument raises `TypeError` instead of binding to whichever option
  happened to sit second in that subclass.
- `blocksfft` no longer takes `use_parallel`. Both branches already called the
  same shared `windowed_block_fft`, so the flag selected nothing. Analyzer
  constructors still accept `use_parallel` for polar weights and SPOD frequency
  work.
- `weighted_second_order` no longer takes `drop_nonpositive`; passing it raises
  `TypeError`. The flag had stopped selecting anything — both routes always drop
  modes at or below their relative cutoff — so a caller passing
  `drop_nonpositive=False` to keep the full spectrum silently received the
  filtered one. The filtering itself is unchanged.
- Welch block partitioning matches `scipy.signal.welch`. Records that do not
  divide evenly change block count. Short records that cannot form one full block,
  and callers requesting more blocks than fit, now raise `ValueError` instead of
  returning empty or wrapped indices. The same floor helper (`welch_nblocks`)
  serves `BaseAnalyzer.load_and_preprocess` and `commands._apply_snapshot_limit`
  after `max_snapshots` truncation, which previously recomputed with ceil and
  could request more blocks than fit. `novlap >= nfft` (hop ≤ 0) is rejected in
  both FFT paths instead of repeating block 0.
- DMD `rank` is required. It no longer defaults to `n_modes_save`, a plotting
  parameter. On the shipped cylinder wake that silent default moved the recovered
  shedding frequency by about 20×, so the library published a number the user
  never chose. There is no principled automatic default either — full numerical
  rank and SVHT were both rejected for fluid spectra. `n_modes_save` now only
  bounds how many modes are kept after sorting.
- BSMD rejects input it cannot analyse instead of returning something plausible.
  A triad component outside the available rfft bins (`|p| > nfft//2`) raises
  `ValueError` naming the index and the bin count, where it previously produced a
  NaN eigenvalue; the last real bin, `|p| = nfft//2`, is still accepted. Dynamic
  triad selection (`use_static_triads=False`) raises `NotImplementedError` instead
  of printing a notice and returning empty arrays. Consequence for small
  transforms: the default triad table reaches `|p| = 8`, so `nfft < 16` with the
  default triads now raises rather than filling high-index rows with NaN.
- `perform_spod()` raises `RuntimeError` when FFT blocks have not been computed.
  It previously printed one line and returned `None`, so callers continued as if
  an analysis had run.
- One rule turns a spatial weight into a vector, replacing three helpers that
  disagreed about which shapes they accepted. A square weight matrix is read as
  its diagonal everywhere — the shape the docstrings always advertised, which
  previously raised `IndexError` from inside a private helper. mPOD now validates
  its weights like every other method. A complex weight array raises on every
  entry path instead of being cast to its real part under a `ComplexWarning`.
- Removed `n_threads` from `BaseAnalyzer`, `SPODAnalyzer` and `blocksfft`. It
  never affected FFT or BLAS work.
- Removed `get_optimization_info()` and `print_optimization_status()`. Both lost
  their callers, and the BLAS name they reported was guessed by string-matching
  NumPy's debug printout, so it went stale whenever that printout changed.
  `get_threadpool_summary()` asks `threadpoolctl` directly and stays.
- Removed the undocumented per-module entry points.
- README documents `FFT_BACKEND` via the public re-export
  `from openmodalpy.core import FFT_BACKEND`.

### Added

- Process-wide BLAS thread policy: `set_blas_threads` / `get_blas_threads` /
  `blas_threads` context manager, and `OPENMODALPY_BLAS_THREADS`. The default of
  1 thread makes `svd`/`eigh`/`eig` reduction order deterministic for a fixed
  environment; `0` means this package applies no limit.
- Result files record a `prov_*` provenance block — versions, FFT backend, BLAS
  threads, config hash, seed, git SHA, UTC timestamp — exposed as
  `AnalysisResults.provenance`.
- `PSDPODAnalyzer`, a library-facing class with the shared analyzer lifecycle.
  The CLI now calls it; numbers and file layout are unchanged. One behavioural
  difference: PSD-POD used to run on an `SPODAnalyzer` and inherit its on-disk
  FFT cache, so it no longer writes a SPOD-named cache beside its results.
- POD `solver` route: `perform_pod(solver="eigh"|"svd")`, config
  `params: {solver: "svd"}`, and `openmodalpy analyze --solver {eigh,svd}`. The
  route was documented but reachable only from a config file.
- Opt-in randomized (Halko) SVD: `randomized_svd` and
  `compute_reduced_svd(..., method="randomized")`. Accuracy tracks spectral decay,
  so `"auto"` never selects it.
- Analyzer argument `spatial_weights=` (type `"prescribed"`), with
  `spatial_weight_type` validated at construction against
  `{"uniform", "polar", "prescribed"}`.
- Config key `energy_fraction` for DMD `rank="energy"` (float in `(0, 1]`; omit
  for the analyzer default `0.999`).
- Analytic reference fixtures under `tests/fixtures/reference/` — POD energy
  fractions and DMD |λ|/phase for `double_gyre`, `taylor_green`, `cylinder_wake` —
  plus `scripts/regen_reference_fixtures.py`, so a clean checkout can recompute
  and check the spectra.
- The three built-in generators are now checked against the closed-form answers
  they already carried. `example_data.py` always returned the double gyre's
  forcing frequency, the Taylor-Green decay eigenvalue and the cylinder wake's
  Strouhal number alongside the data, but nothing read them, and two of the three
  were exercised by no test. Each now runs through the analyzer that should
  recover its quantity, compared against the generator's own metadata rather than
  a constant copied into the test, so the check follows the generator if its
  physics changes. Tolerances come from the discretization: machine precision for
  Taylor-Green, a tenth of the Rayleigh frequency for DMD, half an FFT bin for
  SPOD.
- A test that two prescribed metrics give two different eigenvalues on POD,
  ST-POD, mPOD, SPOD, PSD-POD and BSMD, and the same eigenvalues on DMD — which
  documents that DMD does not use `self.W`.
- SPOD warns when it saves FFT blocks carrying no cache stamp, because the next
  run cannot validate them and must recompute.
- PEP 561 marker `py.typed`, and public exports `MethodInfo` (returned by
  `list_methods` and `get_method_spec`), `ExampleInfo` (returned by
  `discover_examples`) and `RunCollectionSpec`, so callers can annotate results
  without importing from a private module.
- `analyze` help lists method names from the registry instead of a hand-maintained
  string.

### Changed

- Every analyzer reports on its module logger instead of standard output:
  `openmodalpy.pod`, `openmodalpy.psd_pod`, `openmodalpy.spod`,
  `openmodalpy.stpod`, `openmodalpy.bsmd`, `openmodalpy.dmd`. Messages that need
  the user to act — no results file to plot, a mode with no valid data, weights of
  an unexpected shape — carry `WARNING` or `ERROR` level rather than an inline
  `Warning:` prefix.
- Mode sign and phase are canonical: each mode is scaled so the pivot entry — the
  lowest index whose magnitude sits within `CANONICAL_TIE_RTOL = 1e-12` of the
  column maximum — is real and positive. Previously LAPACK's arbitrary sign or
  phase passed straight through, so a different build or thread count could flip a
  published mode shape while every test still passed. The band moves the ambiguity
  threshold, it does not remove it: exactly antisymmetric modes remain a genuine
  tie, and where eigenvalues repeat, any orthonormal basis of that subspace is
  valid. Time coefficients receive the same factor, so reconstruction is
  unchanged. Covers POD (both kernel branches), mPOD, ST-POD and the complex
  PSD-POD route; SPOD and BSMD do not share that seam and are not yet canonical.
- POD, mPOD and PSD-POD share one relative eigenvalue cutoff
  (`λ ≤ n_kernel·ε·λ_max`), so rank-deficient input returns only honest unit-norm
  modes and the count is scale-invariant across unit systems. Eigenvalues are
  unchanged on every recorded fixture; reported energy fractions move in the last
  digit or two because the total no longer includes the noise-level and negative
  eigenvalues the old absolute floor let through.
- POD, mPOD, ST-POD and PSD-POD share one lift / metric / second-order seam
  (`IdentityLift`, `DelayEmbeddingLift`, `BandFilteredLift`, `SpatialMetric`,
  `weighted_second_order`). Results unchanged; each caller keeps its own
  truncation policy via `n_keep`.
- One windowed-block Welch FFT: `blocksfft` and `blocksfft_optimized` both call
  `core/welch.py::windowed_block_fft`. The two copies were already bit-identical.
  Analytical checks pin power-norm Parseval, amplitude recovery and a
  `scipy.signal.welch` cross-check.
- One SPOD single-frequency eigenproblem and one load-latest result search. The
  serial path's modes shift by up to 7e-15 in absolute sum, because dropping a
  diagonal matrix multiply reassociates the same floating-point product;
  eigenvalues are bit-identical.
- The ten `plot_modes_3d_{slices,isometric}` methods share one driver. No figure
  output changed.
- Rendering a 3-D mode figure allocates about one copy of the volume instead of
  three. Measured on a 256×128×128 float64 volume (32 MiB): peak allocation before
  the plotting library is called falls from 96 MiB to 32 MiB. Colour limits are
  unchanged, including for fields containing NaN or infinities.
- The Welch-family analyzers (SPOD, BSMD, PSD-POD) share one FFT cache and can
  adopt one another's blocks when the stamped parameters match, since all three
  produce identical blocks. Each still writes only its own `..._<type>.hdf5`. The
  cache is looked for in the analyzer's own `results_dir`; BSMD previously looked
  in the globally configured directory regardless of where it was writing.
- SPOD `freqs_to_plot` / `modes_to_plot` and BSMD `triad_indices` /
  `static_triads` accept the numpy arrays users already hold. The annotations said
  `Sequence`, which an `ndarray` is not, so a type-checked caller could not pass
  `np.argsort(...)` even though every body already iterated the array at runtime.
  Runtime behaviour is unchanged.
- POD, ST-POD and DMD `load_results` go through `read_results`, so
  pre-unification files with capitalised dataset names load with the reader's
  `DeprecationWarning`.
- `openmodalpy analyze` rejects an unknown method while parsing the command line
  rather than failing later. The accepted set comes from the method registry, so
  it cannot drift, and the error names the available methods. Every spelling that
  worked before still works.
- Reference fixtures use the grids from the packaged example configs (double_gyre
  80×40/Nt=200, taylor_green 64×64/Nt=100, cylinder_wake 100×50/Nt=500, was
  24×12/40, 24×24/40, 32×16/80). Spectrum values move with the grids; ranks and
  tolerances are unchanged. A set-equality test requires every generator to have a
  fixture.
- The packaged `cylinder_wake.jsonc` states `"seed": 42` explicitly. It always
  used that seed as the generator default; the fixture now reads it from the
  config like it already read `Nx`, `Ny` and `Nt`. No generated data changes.
- The bispectral energy map no longer silently discards triads. Its grid was a
  fixed 17×17 centred on `|p| = 8`, so with `nfft=32` a triad at `p=12` was
  computed and then dropped from the map without a word. The half-width now
  follows the triads actually analysed. The default triad list still produces the
  same 17×17 grid with the same values.
- Four simplifications the code made silently are now written down where a user
  would look for them. `spatial_weight_type="uniform"` returns ones rather than
  cell volumes, so reported energy is a sum over mesh points, not a domain
  integral, and its value changes when the grid is refined. mPOD decomposes each
  band independently and concatenates without joint orthonormalization, so the
  pooled mode matrix is not a W-orthonormal basis — measured on a three-band case,
  cross-band inner products reach 0.5 while within-band ones sit at 1e-16. SPOD's
  `dst` is a Strouhal step, `St[1] - St[0] = df·L/U`, not the frequency resolution
  `fs/nfft`, so the characteristic length and velocity rescale the reported
  eigenvalues; the two coincide only at the default `L = U = 1`, which is why the
  shipped generators never revealed it. The default BSMD triad table covers bin
  indices up to `|p| = 8`, which at `nfft=128` is the bottom 12.5 % of the
  spectrum. No arithmetic changed — only what the project says about it.
- The validation suite enforces its claims. `tests/test_all.py` described itself
  as validating correctness against known analytical solutions, but all 22 checks
  reported through a helper that printed a tick and appended to a list, and the
  pass/fail decision lived in a `main()` reachable only by running the file as a
  script — while CI runs pytest. Under pytest the five tests passed
  unconditionally, each wrapped in a bare `except Exception`, so a crash inside
  POD, DMD or SPOD was still reported as a pass. All 22 are now plain assertions
  carrying the measured value, at their original tolerances. The conversion was
  verified by mutation, not by the suite going green: perturbing POD eigenvalues
  by 5 %, DMD eigenvalues by 2 %, or shifting the SPOD spectrum by four bins each
  turns the suite red.

### Fixed

- POD and ST-POD use the spatial metric `load_and_preprocess` built, including on
  the uniform path. POD overwrote `self.W` with ones whenever the type was
  `"uniform"`, and ST-POD's `_get_weight_vector` returned ones regardless, so its
  eigenproblem ignored the metric while the saved file still named it. Numbers are
  unchanged, because both resets only wrote ones over ones once the uniform
  builder started returning one weight per point. A provenance test patches the
  builder to a non-ones column and asserts the eigenvalues move, so a later
  cell-volume metric cannot be discarded in silence.
- Scattered point coordinates are a supported input. When `x` and `y` are 1-D and
  as long as the snapshot matrix is wide — one coordinate pair per column, which
  is what an unstructured mesh gives you — the uniform metric is built with one
  weight per point. Before, it was always a grid tensor product, so a cloud of `n`
  points produced `n*n` weights. `calculate_uniform_weights` gained an optional
  `n_space`; left out, it still returns the tensor product, so existing callers
  are unchanged. SPOD and BSMD accept scattered input now instead of refusing it.
  Grid runs are unchanged, down to the bit.
- `load_results` rejects a results file whose metric `W` does not match the file's
  own spatial size. The four readers called `_as_spatial_weight_column` without
  `n_space`, so a 3-entry `W` beside a 32-point field loaded as `(3, 1)` and the
  analysis ran on a metric that does not belong to the data.
- `self.W` is always a column `(n_space, 1)` — after load, after a run, and after
  a save/load round trip. POD's uniform path used to overwrite that column with a
  flat vector.
- PSD-POD writes the metric `W` into its results file. The eigenproblem already
  used `self.W`, but `save_results` omitted it, so two runs under different
  metrics produced different modes and files whose provenance looked identical.
- Prescribed weights have the same column shape as the uniform and polar builders.
  The prescribed path stored a flat vector, so BSMD's `W * prod` against an
  `(Nspace, Nblocks)` field raised `ValueError` and never ran.
- A metric that is not an inner product is rejected as soon as data is loaded,
  instead of later inside the solver.
- The MAT loader no longer double-counts an absent coordinate. A `.mat` carrying
  `y` and no `x` set `Nx` to the whole snapshot width, so `Nx*Ny*Nz` counted `y`
  twice. An absent axis now contributes extent 1. Analyzers also reject a custom
  loader whose reported grid product disagrees with `q.shape[1]`, naming both
  numbers. Datasets with no grid metadata are left alone.
- DMD no longer amplifies noise into modes when the snapshot pair is
  ill-conditioned. Both the reduced operator and the mode recovery divide by the
  singular values of the first snapshot matrix, and the number kept was whatever
  you asked for rather than whatever the data supports. A rank-deficient sequence
  alone is harmless — the small singular values cancel — but as soon as the second
  snapshot matrix carries content the first cannot represent, which is what a
  transient or a truncated record produces, the division has nothing to cancel
  against. On a rank-3 sequence perturbed at the final snapshot this returned
  eigenvalues of magnitude 6.7e9 and modes of 1.9e9, all finite, so nothing
  raised. Singular values are now kept only above a threshold relative to the
  largest, following the `numpy.linalg.pinv` convention, which makes the cut
  invariant to the overall scale of the data.
- DMD reports the rank it actually used as `effective_rank`, and warns when that
  is below the modes requested — a `RuntimeWarning` about the data, since asking
  for more modes than the data supports is normal.
- DMD `rank="svht"` thresholds with the unknown-noise coefficient
  `omega(beta) = lambda(beta)/sqrt(mu_beta)` (`mu_beta` the Marchenko–Pastur
  median). The previous form used the known-noise `lambda(beta)` against
  `median(s)`, so the threshold sat about 24 % low at `beta = 1` and pure i.i.d.
  noise kept spurious modes at realistic matrix sizes.
- The SVD route drops singular values at or below `n_kernel · ε · σ_max`, the same
  relative scale as the eigh floor. On exactly rank-3 data both routes return 3
  modes; a planted mode at singular-value ratio `1e-10` is still recovered. The
  previous path kept the full numerical null-space tail — eigenvalues ~1e-29
  against a top eigenvalue of hundreds.
- The SVD route keeps every mode the data supports, so it agrees with the
  eigenvalue route. Subtracting the mean costs one snapshot's worth of
  information, so the limit is `min(n_samples - 1, n_space)`. The route used
  `min(n_samples, n_space) - 1`, the same number only when there are at least as
  many grid points as snapshots. With fewer grid points than snapshots it
  discarded a genuine mode — 40 snapshots on 25 points returned 24 modes against
  the eigenvalue route's 25 — and at a single spatial point it collapsed to zero
  modes for any number of snapshots. ST-POD applied the same wrong limit before
  calling the solver; both are corrected.
- POD reports the true energy total. The reported total was the sum of the
  returned eigenvalues, which omitted the last one, so every percentage read
  slightly high — measured 2.5e-3 relative on a 40×25 case. It now comes from the
  exact identity `norm(X_w, 'fro')**2 / m`, independent of how many modes the
  solver is asked for.
- The SVD route no longer returns a meaningless extra mode when a caller centers
  data whose mean dwarfs the fluctuation. The relative singular-value floor stops
  recognising the nulled direction once the removed mean is about a thousand times
  the fluctuation, because subtracting it destroys too many digits. The route now
  measures whether the input is row-centered and tightens the bound only when the
  measurement says so; detection holds to a mean about 1e9 times the fluctuation.
  Callers passing an explicit `n_keep` — which includes POD and ST-POD — are
  unaffected.
- ST-POD returns one more mode in the temporal-lift regime. It capped at
  `min(m - 1, n)`, the bound mean-centering would justify, but the matrix it
  factors is the delay-embedded lift of a centered series, which is not itself
  row-centered and has full row rank. The cap is now `min(m, n)`.
- `n_modes_saved` reports how many modes the file holds, not how many were asked
  for; `load_results` lowers `n_modes_save` to the width of the file it read,
  never raises it, and never drops modes the file holds. POD, ST-POD, multi-band
  mPOD, DMD and PSD-POD all lower the counter when the solver returns fewer modes
  than the cap, including a degenerate DMD that holds none.
- mPOD's one-call path (`MPODAnalyzer.run_analysis`) used to run plain POD and
  write those results into a file named `..._mpod.hdf5`. The orchestrator now
  dispatches through an overridable `_perform_decomposition` hook, and a test
  requires every `PODAnalyzer` subclass to override it, so the next subclass
  cannot inherit the parent's decomposition unnoticed. The CLI was unaffected.
  DMD also gains a minimal `run_analysis`.
- A multi-band mPOD run announces the decomposition it performed. The single-band
  shortcut inherited POD's start, timing and mode-count lines; two or more bands
  ran the multiscale loop in silence.
- An mPOD run says `mPOD` on the console and on its figures. Six lines during a
  run and two more on load said a bare `POD`; the save line printed a path
  containing `mpod` inside a sentence saying `POD`. All ten drawn titles said
  `POD` too, so a multiscale-POD figure could reach print carrying a plain-POD
  label. Titles, banner and summary now read one shared display-name map, which
  keeps the conventional casing for `mPOD`, `PSD-POD` and `ST-POD`.
- POD's energy-captured report no longer always prints 100 %: the fraction is the
  truncated eigenvalue sum over the pre-truncation total, stored as
  `energy_captured_fraction`.
- PSD-POD reports a mode value of exactly `0` at a zero-measure cell, matching the
  real routes. Modes were built from the unweighted Fourier ensemble, so data at a
  masked cell — which the metric says must contribute nothing — appeared as that
  cell's mode value and could become the sign pivot, corrupting the whole column.
  Eigenvalues were always correct; positive-weight cells drift about `1e-15`.
- SPOD no longer reports a roundoff-negative eigenvalue with its sign flipped. The
  cross-spectral matrix is positive semi-definite, so a slightly negative
  eigenvalue is roundoff; taking its absolute value presented it as real energy.
  Such values are clamped to exactly zero.
- SPOD and BSMD modes no longer flip sign or phase between runs.
- A corrupt, truncated or malformed FFT-block cache no longer aborts the analysis.
  Interrupted runs, full disks and killed jobs leave half-written HDF5 caches that
  still exist on disk; opening one for append used to raise and stop SPOD or BSMD
  even though the blocks are re-derivable from the raw data. Write mode is now
  chosen by whether the file is readable as HDF5, not by whether it exists. The
  read guard also covers a file that opens cleanly but holds a wrong-rank
  `FFTBlocks` dataset or an uncastable stamp. Reading a saved *results* file keeps
  the opposite policy and still raises, since results are not re-derivable.
- A write failure while BSMD saves or offloads its cache is no longer reported as
  a cache-load failure; the write error propagates instead of triggering a
  recompute that cannot save either. A genuine read failure now names the file.
- A large BSMD analyzer stays usable after `save_results`. When the FFT blocks are
  too large for memory, BSMD reads them from a cache file, and to write results
  onto that same file `save_results` must close the handle it reads through. It
  closed the handle but still recorded the blocks as available, so any later use —
  a second `perform_bsmd` with different triads, or a read of the bin count —
  reached a file that was no longer open. The handle is reopened once the write
  finishes, including when the write fails.
- BSMD with no FFT blocks loaded says "no frequency bins are loaded" instead of
  quoting a bound of `|p| <= -1`, and `perform_bsmd` raises `ValueError` on an
  empty `qhat` rather than printing a note and continuing into the analysis.
- BSMD default static triads no longer fail a small-`nfft` configuration.
  `static_triads` defaults to `None` and resolves to a private copy of
  `ALL_TRIADS`; when that default is used, triads outside the bin bounds are
  dropped with a warning naming them. A user-supplied list still raises.
- BSMD static-triad validation bounds by both `nfft//2` and the loaded `qhat`
  length, and no longer swallows out-of-range bin reads into a silent NaN
  eigenvalue.
- The sampling rate `fs` fails with a diagnosis instead of an accident. `fs`
  starts at `0.0` until a dataset is loaded, and on paths that never load one —
  reopening saved results, for instance — that zero reached the frequency code,
  where a periodogram rejected it with a message naming nothing and an `rfftfreq`
  axis raised `ZeroDivisionError`. Both now raise a single `ValueError` naming the
  data source and saying what to supply.
- An all-zero or otherwise degenerate field returns empty results with a warning
  instead of failing inside the eigensolver with `LinAlgError: Array must not
  contain infs or NaNs`.
- `compute_reduced_svd` no longer routes near-full-rank requests to ARPACK. The
  gate is `use_iterative_svd(min_dim, rank)`: iterative only when
  `rank < 0.05 * min_dim` and `min_dim >= 256`. Callers asking for `k = n_min - 1`
  stay on dense SVD, the faster path once rank is a large fraction of the smaller
  dimension.
- ARPACK-path SVD is bit-reproducible via a deterministic local start vector. Of
  the synthetic generators, only the cylinder wake accepts a `seed` and records it
  as `data_seed`; `double_gyre` and `taylor_green` are deterministic and take
  none. Tests reseed NumPy from `OMPY_TEST_RNG_JITTER` so collection order cannot
  leak unseeded draws.
- `canonicalize_modes` accepts an integer-dtype `modes` array instead of failing
  with numpy's `UFuncOutputCastingError`. The scale factor is not an integer, so
  integer input is promoted to `float64`; `float` and `complex` keep their dtype
  and `float32` is not promoted.
- `read_results` loads 0-d (scalar) HDF5 datasets into `extra` instead of raising
  on a scalar dataspace.
- `test_prescribed_weights_change_the_eigenvalues` uses an equal-mean weight pair
  (ones against a renormalised off-centre bump), so a solver consulting only
  `mean(W)` no longer passes; the previous ones-against-ramp pair was isospectral
  on that fixture.
- The repo copy of `examples/cylinder.jsonc` restores the per-run `rank: 4` on its
  `hodmd` and `tls_hodmd` runs, matching the packaged config, and the tests now
  compare the full per-run rank mapping so this class of drift fails the suite.
- Config booleans for `rank` and `energy_fraction` raise at parse time instead of
  being silently treated as missing.
- PSD-POD metadata records `uses_mean_subtraction=True`, matching `blocksfft`,
  which always removes a mean. The previous write stored `False`.
- The CLI's unhandled-command fallback returns exit code 2 instead of relying on
  an unreachable line after `parser.error`.
- SPOD `load_and_preprocess` docstring no longer claims parent mean subtraction or
  a `self.data_matrix` attribute that is never assigned.
- DOC.md: the `bsmd.py` filename, the POD branch condition `Ns < Nspace` (no false
  `<<` margin), a stale hardcoded test count, and a note that DMD neither centers
  nor applies the spatial metric.


## [0.3.0] - 2026-07-27

### Breaking
- **The BSMD cross-bispectral matrix now conjugates the sum-frequency term.**
  Earlier releases formed `E[X(f1) X(f2) X(f1+f2)]`, a plain third-order moment.
  The bispectrum takes the conjugate of the sum-frequency component, so the matrix
  is now built as `B = Q_{k+l}^H W (Q_k ∘ Q_l) / N_blk`, where `∘` is the
  elementwise product, with the modes read from the right eigenvector.

  **BSMD results produced by 0.1.0 or 0.2.0 are invalid and must be recomputed.**
  This is not a question of precision. Without the conjugate the phases of the
  three components never cancel, so the eigenvalue reduces to a random walk of
  size `1/sqrt(N_blk)` instead of converging to the bispectral amplitude.
  Eigenvalues, modes, energy maps and every figure derived from them are affected.
  One case is unaffected: for the `(k, -k, 0)` triads the sum frequency falls on
  the DC bin, which is real for a real field, so conjugating it changes nothing and
  those results were correct all along.

  `load_results` refuses an HDF5 file written before the fix, so an old results
  file raises with the reason instead of feeding stale eigenvalues into a new
  analysis in silence.
- **Window convention for blocked FFT / SPOD:** both serial and parallel
  `blocksfft` paths now use the PERIODIC window from
  `scipy.signal.get_window(..., fftbins=True)`. The optimized path previously
  ignored `window_type` other than `"sine"` (falling back to a SYMMETRIC
  `np.hamming`) and silently substituted Hamming for `hann`/`blackman`/etc.
  Default SPOD spectra therefore change for existing users; serial and parallel
  results now agree bit-for-bit for all supported window names.
- An unrecognised `window_type` now raises instead of quietly falling back to
  Hamming. Names the optimized path used to accept by accident — `hanning`, any
  capitalised spelling, and outright typos — are rejected, since only the exact
  names `scipy.signal.get_window` knows (plus `sine`) are valid.
- **Loaders no longer invent a timestep, and neither do the physical quantities that
  consume one.** A `.mat` file with no `dt` used to be given `dt = 1.0` by the loader
  before validation ever saw it, so the check added below passed on a fabricated value.
  The `.mat` loader now leaves `dt` unset, and `_infer_dt_from_times` returns nothing
  rather than `1.0` when the time vector has fewer than two samples or is constant.
  Every physics-bearing consumer — the DMD continuous-time eigenvalues
  `ω = log(λ)/dt`, the mPOD Nyquist frequency that resolves band edges, and the SPOD
  sampling rate restored on reload — now goes through one validated accessor and raises
  when the timestep is missing, zero, negative or non-finite. Reloading results from an
  HDF5 file that carries no `dt` attribute raises instead of reporting growth rates
  computed at a unit timestep. The mPOD case is the one worth re-checking in existing
  work: a wrong `dt` there moved the band edges, so different modes landed in different
  bands rather than merely being mislabelled.
- **Plots stop labelling an axis in seconds when no timestep is known.** The
  time-coefficient plots for POD, ST-POD and DMD used to build their abscissa as
  `arange(n) * 1.0` and label it `Time` regardless. They now fall back to sample
  indices labelled `Sample index`, and use an explicit time vector or a real `dt`
  when one exists. `DMDAnalyzer.plot_eigenspectra` raises without a usable `dt`,
  since frequency and growth rate are that figure's two axes. The DMD mode-field
  plots keep drawing and simply omit the `f=...` fragment from their titles — the
  mode shapes never depended on the timestep.
- A missing or unusable timestep now raises `ValueError` instead of defaulting to
  `0.1` and continuing. This covers `dt` zero, negative or non-finite, as well as
  absent, `None`, or non-scalar — all of which previously either fabricated a
  timestep or leaked a `KeyError`/`TypeError`. Frequency axes, Strouhal numbers,
  and DMD growth rates require a real positive `dt` from the data or loader.

### Changed
- Relicense from MIT to Apache-2.0, effective from 0.3.0 onward. The 0.1.0 and
  0.2.0 releases remain under MIT.

### Fixed
- The blocked-FFT cache is validated against the parameters that produced it.
  Each cache file now records the window type, window normalization, overlap,
  `nfft`, the block preprocessing flags and a digest of the input array, and a
  cached result is reused only when all of them match. The cache was previously
  keyed on the result filename alone, so changing only `window_type` reused blocks
  computed under the old window, and two datasets sharing a data root, `nfft`,
  overlap and snapshot count could serve each other's blocks. Cache files written
  by earlier versions carry no stamp and are recomputed once.

## [0.2.0] - 2026-07-25

### Changed
- **Breaking:** the import name is now `openmodalpy`, matching the PyPI
  distribution name. Update `import modalpy` / `from modalpy import ...` to
  `import openmodalpy` / `from openmodalpy import ...`.
- **Breaking:** the console script is renamed from `modalpy` to `openmodalpy`
  (e.g. `openmodalpy run --config analysis.jsonc`).

## [0.1.0] - 2026-07-24

First public release, distributed on PyPI as `openmodalpy`.

### Added
- Modal decomposition analyzers: POD, mPOD, PSD-POD, SPOD, ST-POD, DMD (LS/TLS),
  HODMD (LS/TLS) and BSMD.
- Configuration-driven workflow: a single JSONC file runs several methods over one
  dataset, via `openmodalpy run --config`.
- Command line interface: `analyze`, `run`, `methods`, `examples`, `results`.
- Data loaders for `.mat` and `.npz` inputs, plus support for user-supplied loaders.
- Bundled self-contained example configs backed by analytic generators
  (`double_gyre`, `cylinder_wake`, `taylor_green`, `run_benchmarks`).

### Changed
- FFT backend dispatch moved out of this package into
  [`fftkit`](https://github.com/openfluids/fftkit), now a required dependency.
  `openmodalpy.core.config.FFT_BACKEND` re-exports the backend fftkit resolves, so the
  reported backend always matches the one actually used.
- The `mkl` and `gpu` extras now defer to `fftkit[mkl]` and `fftkit[gpu]`.

### Removed
- The bundled `openmodalpy.fft` subpackage. Import FFT helpers from `fftkit` instead:
  `get_fft_func`, `periodogram_rfft`, `find_peaks` and related functions.

[Unreleased]: https://github.com/openfluids/openmodalpy/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/openfluids/openmodalpy/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/openfluids/openmodalpy/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/openfluids/openmodalpy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/openfluids/openmodalpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/openmodalpy/releases/tag/v0.1.0
