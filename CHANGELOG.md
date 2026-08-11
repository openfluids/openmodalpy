# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `SPODAnalyzer.plot_eigenvalues_v2` is renamed to `plot_eigenvalues`. The
  `_v2` suffix is gone; SPOD never had another `plot_eigenvalues` to collide
  with. Output figure basenames drop the `_v2` token as well.
- `SPODAnalyzer.plot_time_coeffs` is renamed to `plot_time_coefficients`,
  matching POD, DMD and ST-POD. The SPOD signature
  (`modes_to_plot`, `freq`, `n_blocks`) is unchanged — only the method name.
- `STPODAnalyzer.plot_time_coefficients` renames its `n_coeffs` parameter to
  `n_coeffs_to_plot`, matching POD and DMD. Call sites that used the old
  keyword must switch.
- `blocksfft` no longer takes `use_parallel`. Both branches already called the
  same shared `windowed_block_fft` path, so the flag selected nothing. Pass
  the remaining keyword arguments only. Analyzer constructors still accept
  `use_parallel` for polar weights and SPOD frequency work.
- README documents `FFT_BACKEND` via `from openmodalpy.core import FFT_BACKEND`
  (the public re-export), not `openmodalpy.core.config`.

| Old | New |
| --- | --- |
| `SPODAnalyzer.plot_eigenvalues_v2(...)` | `SPODAnalyzer.plot_eigenvalues(...)` |
| `SPODAnalyzer.plot_time_coeffs(...)` | `SPODAnalyzer.plot_time_coefficients(...)` |
| `STPODAnalyzer.plot_time_coefficients(n_coeffs=...)` | `...plot_time_coefficients(n_coeffs_to_plot=...)` |
| `blocksfft(..., use_parallel=...)` | `blocksfft(...)` (argument removed) |
| `from openmodalpy.core.config import FFT_BACKEND` | `from openmodalpy.core import FFT_BACKEND` |

- `weighted_second_order` no longer takes `drop_nonpositive`. Passing it now
  raises `TypeError`. The flag had stopped selecting anything — both routes
  always drop modes at or below their relative cutoff — so a caller passing
  `drop_nonpositive=False` to keep the full spectrum received the filtered one
  with no warning. Nothing about the filtering changes; only the misleading
  parameter goes. Use `n_keep` to control how many modes are returned.
- Spatial weights are now used exactly, with no absolute `1e-12` floor. The
  weights are a quadrature measure (m³ in 3-D, m² in 2-D), so an absolute floor
  had units it cannot have: a mesh whose cells fall below `1e-12` was silently
  inflated, and the energy spectrum was not linear in the measure — scaling a
  metric by `1e-14` changed the eigenvalues by a factor of about 95. A cell with
  zero measure now contributes nothing to the decomposition and its POD/ST-POD
  mode value is exactly `0`, matching what SPOD and BSMD already did. Before,
  such a cell entered the kernel with weight `1e-12`, so data held there — a
  masked wall value, for instance — leaked into the spectrum; a masked cell
  holding `1e6` shifted the eigenvalues by 199 %.
- `SpatialMetric` now rejects square matrices and 3-D weight arrays with a
  named `ValueError` (shape + `np.diag` fix) instead of flattening them and
  later failing on length. It holds a diagonal metric as a vector; pass
  `np.diag(W)` for a diagonal matrix, or the raw array to the analyzer weight
  path. 1-D, `(n, 1)`, and non-square `(n, k)` inputs are unchanged.
- `perform_spod()` now raises `RuntimeError` when FFT blocks (`qhat`) have not
  been computed. It previously printed one error line and returned `None`, so
  callers could continue as if an analysis had run. Call `compute_fft_blocks()`
  or `run(compute_fft=True)` first.
- Removed the unused `n_threads` parameter from `BaseAnalyzer`, `SPODAnalyzer`,
  and `blocksfft`. It never affected FFT or BLAS work; use the BLAS thread
  policy above for pool control.
- Removed `get_optimization_info()` and `print_optimization_status()` from
  `openmodalpy.core.parallel`. Both lost their callers when the per-module
  command-line entry points were deleted, and the BLAS name they reported was
  guessed by string-matching NumPy's own debug printout, so it went stale
  whenever that printout changed. `get_threadpool_summary()` stays and is
  unaffected — it asks `threadpoolctl` directly and returns the real thread
  counts, and every analysis already logs it before computing FFT blocks.
- Removed the undocumented per-module entry points
  (`python -m openmodalpy.pod`, `python -m openmodalpy.spod`,
  `python -m openmodalpy.dmd`, `python -m openmodalpy.bsmd`,
  `python -m openmodalpy.stpod`). Use `openmodalpy` / `python -m openmodalpy`
  instead.
- An unrecognised `spatial_weight_type` now raises at construction instead of
  being kept as-is. This is a behaviour break: any string other than `"auto"`,
  `"uniform"`, `"polar"` or `"prescribed"` used to fall through to the
  grid-spacing weight path and skip POD's and ST-POD's reset to unit weights,
  so a typo silently changed the metric. Code that relied on that fall-through
  to hand an analyzer its own weight vector should now pass
  `spatial_weights=` instead.
- SPOD result files write the spatial grid once, as `x`/`y`/`z` (matching the
  other producers). The previous duplicate datasets `x_coords`/`y_coords`/
  `z_coords` are no longer written. Files that still carry only the old
  `_coords` spelling continue to load: `read_results` maps them onto the
  canonical `x`/`y`/`z` fields and emits a `DeprecationWarning` naming the
  legacy key. Note that files written by earlier versions carry *both*
  spellings, so reading one now emits that warning where it previously did
  not. The canonical `x`/`y`/`z` still win, so values are unchanged — but code
  that turns warnings into errors will need to filter it.
- One HDF5 result contract for every analyzer. Dataset names are lowercase
  (`modes`, `eigenvalues`, `time_coefficients`, `freq`, `st`, `modes1`,
  `modes2`, …); SPOD no longer writes `Weights` (it writes `W` like the
  others). All `save_results` methods share the signature
  `save_results(self, filename=None)` — SPOD gains `filename`, and BSMD's
  `fname` parameter is renamed to `filename` (call sites that passed
  `fname=` must switch). Writing goes through `openmodalpy.core.results`;
  `read_results(path)` returns a typed `AnalysisResults` and still accepts
  the old capitalised layout with a `DeprecationWarning`. `FFTBlocks` keeps
  its name (FFT cache key, not a downstream result field). SPOD result
  files are written in mode `"w"`; BSMD still appends when the destination
  is the open FFT-cache path so that cache is preserved.
- One rule now turns a spatial weight into a vector, instead of three helpers
  that disagreed about which shapes they accepted (`core/base.py`,
  `core/decomposition.py`, `mpod.py`). Two consequences beyond the deduplication.
  A square weight matrix is read as its diagonal everywhere — the shape the class
  docstrings have always advertised, and which previously raised
  `IndexError: tuple index out of range` from inside a private helper. And mPOD
  now validates its spatial weights like every other method, so a negative or
  zero-measure weight raises instead of passing through. A complex weight
  array raises on every entry path that builds or flattens a spatial metric
  (`require_spatial_metric`, `SpatialMetric`, `_coerce_spatial_weights`,
  `_as_weight_vector`), rather than being cast to its real part under a
  `ComplexWarning`. Weight vectors of the usual shape — a length-`n_space`
  column of positive reals — are unaffected.
- An invalid spatial metric now raises instead of producing a confident answer.
  A non-finite weight, a negative weight, or a metric whose total measure is
  zero raises `ValueError`; an isolated zero among positive weights is still
  accepted, and that cell now contributes nothing (see the exact-measure entry
  above). Results for strictly positive weights are unchanged.

  This closes a real hole: polar weights on a grid whose radial coordinate is
  zero give every annulus an area of `pi*r**2 = 0`, and POD would report an
  energy fraction off that empty metric without complaint. Note the condition
  is `r > 0`, **not** `Ny > 1` — a single radial station at `r > 0` has
  positive measure and is fine.

  Scope: all five named methods plus BSMD. The rule has a single definition
  (`core/base.py::require_spatial_metric`); POD, mPOD, ST-POD and PSD-POD reach
  it through the shared seam, SPOD through `spod_function`, and BSMD checks once
  per analysis. SPOD and BSMD apply weights directly rather than flooring them,
  so an isolated zero there means that cell contributes nothing.
- Welch block partitioning now matches `scipy.signal.welch`: `nblocks` is
  computed with floor arithmetic and the remainder is dropped, rather than
  ceil plus a clamped final block that re-uses samples. Records that do not
  divide evenly therefore change block count (the shipped cylinder_wake
  example with `Ns=500`, `nfft=128`, `overlap=0.5` goes from 7 blocks to 6),
  so SPOD / PSD-POD / BSMD numbers on those records move. Short records that
  cannot form one full block, and callers that request more blocks than fit,
  now raise `ValueError` instead of returning empty or wrapped indices.
  The same floor helper (`welch_nblocks`) is used by
  `BaseAnalyzer.load_and_preprocess` and by `commands._apply_snapshot_limit`
  after `max_snapshots` truncation — the snapshot-limit path previously
  recomputed `nblocks` with ceil and could request more blocks than fit
  (e.g. Ns=400, nfft=128, overlap=0.5 → floor 5 vs ceil 6). `novlap >= nfft`
  (hop ≤ 0) is rejected in both FFT paths instead of repeating block 0.
- DMD `rank` is required. `DMDAnalyzer` no longer defaults the
  operator truncation to `n_modes_save` (a plotting parameter). Omitting `rank`
  raises `ValueError` naming the alternatives: a positive `int`, `"svht"`, or
  `"energy"`. On the shipped cylinder wake that silent default moved the recovered
  shedding frequency by ~20×, so a library that picks the rank silently publishes
  a number the user never chose. There is no principled automatic default either
  (full numerical rank and SVHT were both rejected for fluid spectra). Migrate
  by setting `rank` to the value you previously relied on via `n_modes_save`
  (`rank=n_modes_save` is bit-identical to the old default). `n_modes_save` only
  bounds how many modes are kept after sorting.
- BSMD now rejects input it cannot analyse instead of returning something plausible.
  A triad component outside the available rfft bins (`|p| > nfft//2`) raises `ValueError`
  naming the offending index and the bin count, where it previously produced a NaN
  eigenvalue; the last real bin, `|p| = nfft//2`, is still accepted. Dynamic triad
  selection (`use_static_triads=False`) raises `NotImplementedError` instead of printing
  a notice and returning empty arrays. Note the consequence for small transforms: the
  default triad table reaches `|p| = 8`, so `nfft < 16` combined with the default triads
  now raises rather than filling the high-index rows with NaN.

### Added
- `openmodalpy analyze` accepts `--solver {eigh,svd}` and forwards it to POD.
  The solver route was documented under `methods show pod` but reachable only
  from a config file; the CLI flag closes that gap. Omitting the flag leaves
  the library default in charge.
- SPOD warns when it saves FFT blocks that carry no cache stamp. The stamp is
  derived from the source snapshots, so a save without them in memory leaves
  blocks the next run cannot validate and must recompute. The behaviour is
  unchanged; the run now says why.
- Opt-in randomized (Halko) SVD: `randomized_svd` and
  `compute_reduced_svd(..., method="randomized")`. Accuracy tracks spectral
  decay, so `"auto"` never selects it.
- Analyzer argument `spatial_weights=` (type `"prescribed"`) and construction-time
  validation of `spatial_weight_type` to `{"auto", "uniform", "polar", "prescribed"}`.
- Case config key `energy_fraction` for DMD `rank="energy"` (float in `(0, 1]`;
  omit to keep the analyzer default `0.999`).
- POD `solver` route: `perform_pod(solver="eigh"|"svd")` and config
  `params: {solver: "svd"}` select the correlation-matrix (`eigh`, default)
  or weighted-snapshot-SVD path. Documented in DOC.md under non-positive
  eigenvalues / rank-deficient input. CLI help comes from
  `METHOD_REGISTRY["pod"].parameter_help`.
- PEP 561 marker `src/openmodalpy/py.typed` so type checkers treat the package
  as typed when installed from the wheel.
- Public exports `MethodInfo` (returned by `list_methods` and `get_method_spec`),
  `ExampleInfo` (returned by `discover_examples`) and `RunCollectionSpec`, so
  callers can annotate these results without importing from a private module.
- The `analyze` help text now lists the method names from the method registry
  instead of a hand-maintained string that could fall out of date.
- Analytic reference fixtures under `tests/fixtures/reference/` (POD energy
  fractions and DMD |λ|/phase for `double_gyre`, `taylor_green`,
  `cylinder_wake`) plus `scripts/regen_reference_fixtures.py` and
  `tests/test_reference_fixtures.py` so a clean checkout can recompute and
  check the spectra.
- `PSDPODAnalyzer` — library-facing PSD-POD class with the shared analyzer
  lifecycle (`load_and_preprocess` → `compute_fft_blocks` → `perform_psd_pod`
  → `save_results`). The CLI/config path now calls this class; numbers and
  result-file layout are unchanged. One behavioural difference: PSD-POD used to
  run on an `SPODAnalyzer` and so inherited its on-disk FFT-block cache. It now
  recomputes the blocks and no longer writes a SPOD-named cache file beside its
  results. Results are identical; a repeated run on the same large record no
  longer reuses a cached `qhat`.
- Process-wide BLAS thread policy (`openmodalpy.set_blas_threads` /
  `get_blas_threads` / `blas_threads` context manager; env
  `OPENMODALPY_BLAS_THREADS`). Default is 1 thread so `svd`/`eigh`/`eig`
  reduction order is deterministic for a fixed environment; `0` means this
  package applies no limit (outer env / limiters still apply).
- Result files record a `prov_*` provenance block (versions, FFT backend, BLAS
  threads, config hash, seed, git SHA, UTC timestamp) via `write_results`;
  `AnalysisResults.provenance` exposes it with the prefix stripped.
- The three built-in synthetic generators are now checked against the closed-form
  answers they already carry. `example_data.py` has always returned the double gyre's
  forcing frequency, the Taylor-Green decay eigenvalue and the cylinder wake's Strouhal
  number alongside the data, but nothing read them, and two of the three generators were
  not exercised by any test. Each is now run through the analyzer that should recover its
  quantity, comparing against the value read from the generator's own metadata rather
  than a constant copied into the test, so the check follows the generator if its physics
  changes. Tolerances are computed from the discretization: machine precision for
  Taylor-Green, where the field is rank-1 in space times a pure exponential and DMD
  recovers the multiplier exactly; a tenth of the Rayleigh frequency for DMD, which is
  not bin-limited; half an FFT bin for SPOD, which is. A cross-analyzer check pins that
  DMD and SPOD agree on the shedding frequency without reference to the metadata.

### Changed
- POD energy percentages now report a share of total field energy (pre-truncation),
  so regenerated figures will read lower than before for any truncated spectrum.
- mPOD figure files now use `_mpod_` in their names (from `analysis_type`) instead of the hard-coded `_pod_` inherited from PODAnalyzer. Anything that still looks for the old `_pod_` figure names after an mPOD run must update. POD figure names are unchanged.
- `openmodalpy analyze` rejects an unknown method when it parses the command
  line, instead of accepting it and failing later. The accepted set is derived
  from the method registry, so it cannot drift from the methods that exist, and
  the error names them: `Unknown method 'psdpod'. Available: ['bsmd', 'dmd',
  'hodmd', 'mpod', 'pod', 'psd_pod', 'spod', 'stpod', 'tls_hodmd']`. Every
  spelling that worked before still works — `pod`, `POD`, `psd-pod`, `psd_pod`,
  `tls-hodmd` — since the name is normalized before it is checked.
- The packaged `cylinder_wake.jsonc` states `"seed": 42` explicitly. It always
  used that seed, as the generator default, but the value was implicit in the
  shipped config while the repository copy stated it. The reference fixture now
  reads the seed from the config like it already read `Nx`, `Ny` and `Nt`,
  rather than from a separate hardcoded literal, so the fixture's generation
  contract is fully derived. No generated data changes.
- PSD-POD reuses cached FFT blocks again instead of recomputing them on every
  run. The Welch-family analyzers (SPOD, BSMD, PSD-POD) now share one cache
  implementation, and any of them can adopt another's cached blocks when the
  stamped FFT parameters match, since all three produce identical blocks for
  the same parameters. Each still writes only its own `..._<type>.hdf5` file.
- The FFT cache looks for reusable blocks in the analyzer's own `results_dir`.
  BSMD previously looked for an SPOD cache in the globally configured results
  directory regardless of where it was writing; with default settings the two
  are the same directory, so only setups using per-analysis directories see a
  difference.
- FFT cache progress messages (`Loaded cached FFT blocks ...`, `Saved FFT
  blocks to cache ...`) now go to the logger instead of standard output.
- Every analyzer — POD, PSD-POD, SPOD, ST-POD, BSMD and DMD — now reports
  progress, results and diagnostics on its module logger (`openmodalpy.pod`,
  `openmodalpy.psd_pod`, `openmodalpy.spod`, `openmodalpy.stpod`,
  `openmodalpy.bsmd`, `openmodalpy.dmd`) instead of writing to
  standard output. Nothing in the library writes to stdout any more. The
  command-line tool installs a handler and still shows
  every message, so its output is unchanged; a library caller now sees nothing
  on stdout and can route, filter, or silence the messages like any other
  Python logging. Messages that need the user to act — no results file to plot,
  a mode with no valid data, weights of an unexpected shape — carry `WARNING`
  or `ERROR` level rather than an inline `Warning:` prefix.
- Rendering a 3D mode figure now allocates about one copy of the mode volume
  instead of three. `subset_volume_focus_3d` no longer copies the volume when
  no cropping is configured, and `get_robust_clim` no longer copies the data to
  filter it when every value is already finite. Measured on a 256x128x128
  float64 volume (32 MiB): peak allocation before the plotting library is
  called falls from 96 MiB to 32 MiB. Colour limits are unchanged, including
  for fields containing NaN or infinities.
- POD, ST-POD and DMD `load_results` now go through `read_results`, so
  pre-unification files with capitalised dataset names (`Modes`, `Eigenvalues`,
  `TimeCoefficients`) load and emit the reader's `DeprecationWarning`.
- ST-POD percentages now mean share of total field energy (pre-truncation) and
  will read lower than before for any truncated spectrum.
- Default DMD rank for the shipped `double_gyre` example moved from 10 to 8.
  On the packaged 80×40/Nt=200 grid, rank 10 keeps singular values with
  s9/s0 ~ 2e-12, so machine round-off in the DMD operator is amplified to
  ~1e-4 — a hundred times looser than the fixture `rtol=1e-6`. Rank 8 keeps
  s7/s0 ~ 4e-9 (implied error ~5e-8) and is honest at that tolerance. Only the
  DMD family is affected: `dmd`, `hodmd` and `tls-hodmd` runs on the shipped
  config now return 8 modes instead of 10, matching the reference fixture. POD,
  mPOD, SPOD, PSD-POD, ST-POD and BSMD are unchanged — they read `n_modes_save`,
  which stays at 10, not `rank`.
- Analytic reference fixtures under `tests/fixtures/reference/` now use the
  grids from the packaged example configs (`src/openmodalpy/examples/*.jsonc`):
  double_gyre 80×40/Nt=200, taylor_green 64×64/Nt=100, cylinder_wake
  100×50/Nt=500 (was 24×12/40, 24×24/40, 32×16/80). Spectrum values move with
  the grids; ranks and tolerances are unchanged. The regen script and drift
  test both read those grids via `openmodalpy.config_io.load_jsonc` (single
  source; the full generation contract including `seed` is pinned). A set-
  equality test requires every generator to have a fixture.
- One windowed-block Welch FFT: `blocksfft` and `blocksfft_optimized` both
  call `core/welch.py::windowed_block_fft`. Same numbers (the two copies were
  already bit-identical); the loop, `get_window`, and `(cw / nfft)` scaling
  live in one place. Analytical checks in `tests/test_welch_analytical.py`
  pin power-norm Parseval, amplitude recovery, and a scipy.signal.welch
  cross-check. Public signatures and the `use_parallel` branch are unchanged.
- POD, mPOD and PSD-POD share one relative eigenvalue cutoff
  (`λ ≤ n_kernel·ε·λ_max`) on the correlation matrix, so rank-deficient input
  returns only honest unit-norm modes and the count is scale-invariant across
  unit systems. The previous absolute `1e-12` energy floor is gone; the basis
  is not padded when fewer modes are supported than requested. Eigenvalues
  themselves are unchanged on every recorded fixture, but reported energy
  fractions move in the last digit or two: the total they are normalised by no
  longer includes the noise-level and negative eigenvalues that the old floor
  let through, so `energy_captured_fraction` and the per-mode fractions shift by
  around 1e-16 relative. Rank-deficient cases also return fewer modes, because
  the noise tail is no longer reported as modes. Reference fixtures were updated
  for both. No physical result changes.
- The ten `plot_modes_3d_{slices,isometric}` methods share one driver in
  `core/base.py`; each analyzer keeps a private helper for mode selection and
  titles. No figure output changed.
- One SPOD single-frequency eigenproblem and one load-latest result search.
  The serial path in `spod_function` and `spod_single_frequency_optimized` both
  call `core/decomposition.py::spod_single_frequency` (union of `num_modes` and
  `return_psi`; modes via the broadcast form, not `@ np.diag`). Before the
  merge the two copies already agreed to machine zero, and eigenvalues are
  bit-identical after it. The serial path's modes shift by up to 7e-15 in
  absolute sum, because dropping the diagonal matrix multiply reassociates the
  same floating-point product; no other quantity changed. The six load-latest
  auto-detect blocks now call
  `core/results.py::find_latest_result`; each caller still owns its not-found
  policy (mpod silent; the others print `[Auto-detect]` / `[ERROR]`). Net
  `src/` line delta: −13.
- Mode sign and phase are now canonical: each mode is scaled so the pivot
  entry — the lowest index whose magnitude sits within a relative band
  (`CANONICAL_TIE_RTOL = 1e-12`) of the column maximum — is real and positive.
  Near-equal opposite peaks no longer flip under single-ulp noise between builds.
  Previously LAPACK's arbitrary sign (real) or phase (complex) passed straight
  through, and a different LAPACK build or thread count could flip a published
  mode shape while every test still passed.

  The band moves the ambiguity threshold; it does not remove it. Peaks that
  differ by more than the band still decide the sign, and exactly antisymmetric
  modes (`phi` vs `-phi`) remain a genuine tie that any comparison must break
  somewhere. Where eigenvalues are repeated, any orthonormal basis of that
  subspace is a valid answer; fixing each mode's phase cannot make the basis
  itself unique. Non-finite mode entries, and a `coeffs` column count that does
  not match `modes`, raise `ValueError`.

  Time coefficients receive the same factor as the modes, so coefficients stay
  the projection of the data onto the modes and reconstruction is unchanged on
  every route (real: `coeffs @ modes.T`; complex: `coeffs @ modes.conj().T`).
  Eigenvalues and mode subspaces are identical to before. Only the sign
  convention is new.

  Covers POD (both kernel branches), mPOD, ST-POD and the complex PSD-POD
  route. SPOD and BSMD do not share that seam and are not yet canonical.
- POD, mPOD, ST-POD and PSD-POD now share one lift / metric / second-order
  seam in `core/decomposition.py` (`IdentityLift`, `DelayEmbeddingLift`,
  `BandFilteredLift`, `SpatialMetric`, `weighted_second_order`). Results are
  unchanged; each caller keeps its own truncation policy via `n_keep`.
- Four simplifications the code made silently are now written down where a user would
  look for them. `spatial_weight_type="uniform"` returns ones rather than cell volumes,
  so reported POD/SPOD energy is a sum over mesh points, not a domain integral, and its
  numerical value changes when the grid is refined. mPOD decomposes each band
  independently and then concatenates and re-sorts the modes with no joint
  orthonormalization, so the pooled mode matrix is not a W-orthonormal basis even though
  each band's modes are; measured on a three-band case, cross-band inner products reach
  0.5 while within-band ones sit at 1e-16. SPOD's `dst` is a Strouhal step,
  `St[1] - St[0] = df·L/U`, not the frequency resolution `fs/nfft`, so the characteristic
  length and velocity rescale the reported eigenvalues; the two coincide only at the
  default `L = U = 1`, which is why the shipped generators never revealed it. The default
  BSMD triad table covers frequency-bin indices up to `|p| = 8`, which at the default
  `nfft=128` is the bottom 12.5% of the spectrum. A docstring in `core/base.py` that
  claimed the opposite about `dst` has been corrected. None of the underlying arithmetic
  changed; only what the project says about it.
- The bispectral energy map no longer silently discards triads. Its grid was a fixed
  17×17 centred on `|p| = 8`, so a triad outside that window was computed and then
  dropped from the map without a word — with `nfft=32`, where 16 bins are available, a
  triad at `p=12` vanished. The half-width is now derived from the triads actually
  analysed, and the plot extent follows it. The default triad list still produces the
  same 17×17 grid with the same values.
- The validation suite now enforces its claims. `tests/test_all.py` describes itself as
  validating mathematical correctness against known analytical solutions, but every one
  of its 22 checks reported through a helper that printed a tick and appended to a list;
  the pass/fail decision lived in a `main()` reachable only by running the file as a
  script, while CI runs pytest. Under pytest the five tests passed unconditionally, and
  each was wrapped in a bare `except Exception`, so a crash inside POD, DMD or SPOD was
  still reported as a pass. All 22 checks are now plain assertions with the measured
  value in the failure message, at their original tolerances, and analyzer output is
  routed to pytest's `tmp_path` instead of a `./results` directory in the working tree.
  The conversion was verified by mutation, not by the suite going green: perturbing POD
  eigenvalues by 5%, DMD eigenvalues by 2%, or shifting the SPOD spectrum by four
  frequency bins each turns the suite red.

### Fixed
- An mPOD run says `mPOD` on the console. Six lines printed during a run, and
  two more when results are loaded, said a bare `POD`, because the analyzer
  mPOD builds on hardcoded the word. The save line was the clearest symptom: it
  printed a path already containing `mpod` inside a sentence saying `POD`. The
  label now follows the analysis type, the same way the figure titles do. A
  plain POD run is unchanged, and messages naming the `perform_pod()` method
  keep that name, since it is the method both classes really use.
- The SVD route keeps every mode the data supports, so it now agrees with the
  eigenvalue route. Subtracting the mean costs one snapshot's worth of
  information, so the limit on how many modes the data can support is
  `min(n_samples - 1, n_space)`. The SVD route instead used
  `min(n_samples, n_space) - 1`, which is the same number only when there are
  at least as many grid points as snapshots. With fewer grid points than
  snapshots it discarded one genuine mode: 40 snapshots on 25 points returned
  24 modes where the eigenvalue route returned 25, and a run asking for more
  reported the contradictory `n_modes_save (25) > available modes (24)`. At a
  single spatial point the old limit collapsed to zero, so the SVD route
  returned no modes at all for any number of snapshots. The same wrong limit
  was applied a second time by ST-POD before it called the solver, so both
  places are corrected. Results are unchanged wherever there are at least as
  many grid points as snapshots.
- POD reports the true energy total, and keeps every mode the data supports.
  Two separate errors, both on the `solver="svd"` route: the reported total was
  the sum of the returned eigenvalues, which omitted the last one, so every
  energy percentage read slightly high (measured 2.5e-3 relative on a 40x25
  case). The total now comes from the exact identity `norm(X_w, 'fro')**2 / m`,
  which does not depend on how many modes the solver is asked for. Separately,
  POD now asks the solver only for the modes it keeps instead of nearly the full
  rank. The bound for that request is `min(n_samples - 1, n_space)`: subtracting
  the mean costs one snapshot's worth of information, not one of whichever
  dimension is smaller. Fields with fewer grid points than snapshots keep the
  mode that the old bound discarded.
- An mPOD run now labels its figures `mPOD`. Every title drawn inside the
  image said `POD`, because the ten title strings were hardcoded in the POD
  analyzer that mPOD builds on. The filename already said `mpod`, so a
  multiscale-POD figure could reach print carrying a plain-POD label. Titles,
  the start banner and the run summary now all read their name from one
  shared display-name map, which keeps the conventional casing for `mPOD`,
  `PSD-POD` and `ST-POD`. Labels only — no numerical result changes.
- PSD-POD (the complex solver route) now reports a mode value of exactly `0`
  at a zero-measure cell, matching the real POD/ST-POD routes. Modes were
  built from the unweighted Fourier ensemble, so data held at a masked cell —
  which the metric says must contribute nothing — appeared as that cell's mode
  value and could become the sign/phase pivot, corrupting the whole mode
  column. Eigenvalues were always correct; mode values at positive-weight
  cells are unchanged (measured drift about `1e-15`).
- The repo copy of `examples/cylinder.jsonc` restores the per-run `rank: 4`
  on its `hodmd` and `tls_hodmd` runs, matching the packaged config. A
  checkout and an installed wheel now resolve the same rank for every run;
  the example-config tests now compare the full per-run rank mapping, so
  this class of drift fails the suite instead of passing silently.
- `compute_reduced_svd` no longer routes near-full-rank requests to ARPACK.
  The gate is now `use_iterative_svd(min_dim, rank)`: iterative only when
  `rank < 0.05 * min_dim` and `min_dim >= 256`. Callers that ask for
  `k = n_min - 1` (POD SVD route, ST-POD) stay on dense SVD, which is the
  faster path once rank is a large fraction of the smaller dimension.
- `canonicalize_modes` accepts an integer-dtype `modes` array instead of failing
  with numpy's `UFuncOutputCastingError`. The scale factor it applies is not an
  integer, so integer input is now promoted to `float64` before scaling; `float`
  and `complex` inputs keep their own dtype, and `float32` is not promoted. The
  returned arrays were already copies, so a caller's array is untouched either
  way. No solver route was affected — all of them pass float or complex — so
  this only matters when calling the function directly.
- SPOD no longer reports a roundoff-negative eigenvalue with its sign flipped.
  The cross-spectral matrix is positive semi-definite, so an eigenvalue that
  comes back slightly negative is roundoff; taking its absolute value presented
  it as real energy. Such values are now clamped to exactly zero. Only the
  roundoff tail of the spectrum is affected, and the number of returned modes
  is unchanged.
- A malformed FFT cache file now recomputes the blocks instead of raising. The
  read guard previously caught only unreadable files, so a file that opened
  cleanly but held a wrong-rank `FFTBlocks` dataset or an uncastable stamp
  attribute aborted the run. This matters more now that an analyzer may open a
  file written by another analysis.
- `read_results` now loads 0-d (scalar) HDF5 datasets into `extra` instead of
  raising on a scalar dataspace; normal datasets in the same file are unchanged.
- SPOD and BSMD modes no longer flip sign or phase between runs.
- The CLI's internal unhandled-command fallback now returns exit code 2 instead of
  relying on an unreachable line after ``parser.error``.
- Config booleans for `rank` and `energy_fraction` now raise at parse time instead
  of being silently treated as missing (`null`).
- DMD `rank="svht"` now thresholds with the unknown-noise coefficient
  `omega(beta) = lambda(beta)/sqrt(mu_beta)` (`mu_beta` = Marchenko–Pastur
  median). The previous form used the known-noise `lambda(beta)` against
  `median(s)`, so the threshold sat ~24% low at `beta = 1` and pure i.i.d.
  noise kept spurious modes at realistic matrix sizes.
- The SVD route of `weighted_second_order` now drops singular values at or
  below `n_kernel · ε · σ_max` (same relative scale as the eigh floor, applied
  in the singular-value domain). On exactly rank-3 data both routes return 3
  modes; a planted mode at singular-value ratio `1e-10` is still recovered.
  The previous SVD path kept the full numerical null-space tail (eigenvalues
  ~1e-29 against a top eigenvalue of hundreds). ST-POD is the existing SVD
  caller and may return fewer trailing noise modes on rank-deficient input.
- mPOD's one-call path (`MPODAnalyzer.run_analysis`) used to run plain POD and
  write those results into a file still named `..._mpod.hdf5`. The orchestrator
  now dispatches through an overridable `_perform_decomposition` hook
  (`perform_pod` for POD, `perform_mpod` for mPOD), and a test requires every
  `PODAnalyzer` subclass to override it, so the next subclass cannot inherit the
  parent's decomposition unnoticed. The CLI was unaffected: `commands.py`
  already called `perform_mpod` by name. DMD also gains a minimal `run_analysis`
  (load, compute, save; no plots).
- BSMD with no FFT blocks loaded now says so — "no frequency bins are loaded" —
  instead of quoting a bound of `|p| <= -1`, and `perform_bsmd` raises
  `ValueError` on an empty `qhat` rather than printing a note and continuing
  into the analysis.
- A full disk (or other write failure) while BSMD saves or offloads its FFT
  block cache is no longer reported as a cache-load failure; the write error
  propagates instead of triggering a recompute that cannot save either. When a
  cache read genuinely does fail, the message now names the file it could not
  read.
- PSD-POD result metadata now records `uses_mean_subtraction=True`, matching
  `blocksfft` (which always removes a mean — global by default, per-block when
  `blockwise_mean` is set). The previous write stored `False`.
- SPOD `load_and_preprocess` docstring no longer claims parent mean subtraction
  or a `self.data_matrix` attribute that is never assigned.
- DOC.md: `bsmd.py` filename, POD branch condition `Ns < Nspace` (no false
  `<<` margin), dropped a stale hardcoded test count, and notes that DMD neither
  centers nor applies the spatial metric.
- BSMD default static triads no longer fail a small-`nfft` configuration.
  `static_triads` defaults to `None` and resolves to a private copy of
  `ALL_TRIADS`; when that default list is used, triads outside `|p| <= nfft//2`
  (and the loaded-bin bound) are dropped with a warning that names them. A
  user-supplied list still raises `ValueError`, and every out-of-range
  component is named in one message.
- BSMD static-triad validation bounds by both `nfft//2` and the loaded `qhat`
  length, and no longer swallows out-of-range bin reads into a silent NaN
  eigenvalue. The two bounds coincide for a freshly computed transform; when
  they diverge, the triad is now rejected with a `ValueError` naming the real
  bound instead of returning `NaN` with no diagnostic.
- POD energy-captured report no longer always prints 100%: the fraction is
  truncated eigenvalue sum over the pre-truncation total, stored as
  `energy_captured_fraction` on the analyzer and in result metadata.
- ARPACK-path SVD (`compute_reduced_svd` with `min_dim >= 256`) is bit-reproducible
  via a deterministic local start vector. Of the synthetic generators, only the
  cylinder wake accepts a `seed` and records it into result metadata as
  `data_seed`; the JetLES-like dummy generator accepts a `seed` for its noise RNG
  but does not surface it; `double_gyre` and `taylor_green` are deterministic and
  take no seed. Tests reseed NumPy from `OMPY_TEST_RNG_JITTER` so collection
  order cannot leak unseeded draws.
- DMD no longer amplifies noise into modes when the snapshot pair is ill-conditioned.
  The reduced operator and the mode recovery both divide by the singular values of the
  first snapshot matrix, and the number kept was whatever you asked for rather than
  whatever the data supports. A rank-deficient sequence alone is harmless — the small
  singular values cancel — but as soon as the second snapshot matrix carries content the
  first one cannot represent, which is what a transient, an arriving structure or a
  truncated record produces, the division has nothing to cancel against. On a rank-3
  sequence with a perturbation applied to the final snapshot this returned eigenvalues of
  magnitude 6.7e9 and modes of magnitude 1.9e9, all finite, so nothing raised. Singular
  values are now kept only above a threshold relative to the largest one, following the
  `numpy.linalg.pinv` convention, which makes the cut invariant to the overall scale of
  the data; both `pinv` calls pass that same conditioning explicitly instead of relying
  on a default. Well-conditioned data is unaffected.
- DMD reports the rank it actually used as `effective_rank`, and warns when that is below
  the number of modes requested. Asking for more modes than the data supports is normal,
  so this is a `RuntimeWarning` about the data rather than an error.
- An all-zero or otherwise degenerate field returns empty results with that warning
  instead of failing inside the eigensolver with `LinAlgError: Array must not contain
  infs or NaNs`.
- A corrupt or truncated FFT-block cache no longer aborts the analysis. Interrupted
  runs, full disks, and killed jobs can leave a half-written HDF5 cache that still
  exists on disk; opening it for append used to raise and stop SPOD or BSMD even though
  the blocks are re-derivable from the raw data. Write mode is now chosen by whether the
  file is actually readable as HDF5, not by whether it exists, so an unreadable cache is
  overwritten after a recompute. The same recovery applies when BSMD tries to reuse a
  SPOD cache that turns out to be truncated: it prints a reason and recomputes rather
  than raising. Reading a saved results file keeps the opposite policy and still raises,
  since results are not re-derivable from the raw data the way FFT blocks are.
- The sampling rate `fs` fails with a diagnosis instead of an accident. `fs` starts at
  `0.0` until a dataset is loaded, and on paths that never load one — reopening saved
  results, for instance — that zero used to reach the frequency code, where a periodogram
  rejected it with a message naming nothing and an `rfftfreq` axis raised
  `ZeroDivisionError`. Both now raise a single `ValueError` naming the data source and
  saying what to supply, matching the message the timestep already used. Frequency axes
  are unchanged whenever the sampling rate is valid.

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

[Unreleased]: https://github.com/openfluids/openmodalpy/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/openfluids/openmodalpy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/openfluids/openmodalpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/openmodalpy/releases/tag/v0.1.0
