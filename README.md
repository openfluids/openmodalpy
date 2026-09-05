![openmodalpy banner](https://raw.githubusercontent.com/openfluids/openmodalpy/main/assets/readme-banner-v3.jpg)

[![CI](https://github.com/openfluids/openmodalpy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/openfluids/openmodalpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openmodalpy.svg)](https://pypi.org/project/openmodalpy/)
[![Python](https://img.shields.io/pypi/pyversions/openmodalpy.svg)](https://pypi.org/project/openmodalpy/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`openmodalpy` loads your snapshots once and runs every modal decomposition on
them, so you can compare. Nine methods for spatiotemporal data sit behind one
interface: energy-ranked POD modes, frequency-resolved SPOD modes, DMD
eigenvalues with growth rates, nonlinear BSMD triads. One data contract feeds
all of them, one config file runs any subset, and every method writes the same
kind of result file. It runs on NumPy and SciPy. There is no compiled solver
toolchain to install.

## Who this is for

**You have snapshots and need a decomposition you can defend.** From a solver,
an experiment, or a public dataset. Every method here is checked against cases
with a known answer, and every result file records how it was made. See
[How we know it is right](#how-we-know-it-is-right).

**You have a new decomposition and want to try it on real data.** Subclass one
base class, write the maths, and get loading, spatial weights, saving,
reloading, provenance and plots for free. The cost of adding a method is
measured, not guessed. See [Add your own method](#add-your-own-method).

**You are learning what these methods do.** The bundled examples generate their
own data with the answer known in closed form, so you can see a method get the
shedding frequency right, or wrong, before you trust it on your own flow. See
[A path through the examples](#a-path-through-the-examples).

## Why one framework

Most Python tools here specialise. [PyDMD](https://github.com/PyDMD/PyDMD)
covers DMD variants in depth, [PySPOD](https://github.com/MathEXLab/PySPOD)
covers SPOD, [MODULO](https://github.com/mendezVKI/MODULO) covers multiscale
POD. That depth is real. If you only ever need one method, they are excellent
choices.

Modal decompositions fall into three families. You can split them by the
question they answer, by what they do to the snapshots, or by the statistic
they use. All three splits give the same three groups:

| Ask | Family | What it does to the snapshots | Statistic | Methods here |
|---|---|---|---|---|
| Which structures dominate? | Energy (POD family) | diagonalizes the snapshot correlation | second-order | POD, mPOD, PSD-POD, SPOD, ST-POD |
| How do they evolve? | Dynamics (DMD family) | fits a linear map from each snapshot to the next | linear operator | DMD, HODMD, TLS-HODMD |
| How do they feed each other? | Interactions (bispectral) | correlates triads of frequencies | third-order | BSMD |

**Energy.** The POD family diagonalizes the correlation of the snapshots, so
its modes are orthogonal and ranked by the energy they carry. Time enters only
as an ensemble (POD), a frequency band (SPOD, PSD-POD, mPOD) or a delay window
(ST-POD). Nothing assumes the flow obeys a rule.

**Dynamics.** The DMD family fits one linear map that takes each snapshot to
the next. The eigenvalues of that map give every mode a frequency and a growth
rate. The modes are not orthogonal and not ranked by energy, because that is
not what they are for.

**Interactions.** Bispectral mode decomposition looks for triples of
frequencies where two combine into a third. Second-order statistics cannot
see this, so the first two families cannot either.

A modal analysis usually asks these three questions in that order, on the same
data. That is what the load-once loop below is for. The split is not perfectly
clean. SPOD resolves frequency, so it says something about dynamics, and on
statistically stationary data it agrees with DMD. DMD modes carry amplitudes,
so they say something about energy. The labels name what each family is built
to answer, not everything it can tell you.

No other open package spans all three families behind one analyzer lifecycle,
one data contract and one result format. That span is the reason this package
exists. Running POD, SPOD, DMD and BSMD over one dataset and comparing them is
a loop in Python or one command line, not four integrations. Bispectral mode
decomposition in particular has little open-source coverage elsewhere.

## Installation

```bash
uv add openmodalpy                 # library
uv tool install openmodalpy        # standalone CLI
```

Optional extras:

| Extra | Adds |
|-------|------|
| `openmodalpy[viz3d]` | 3D slice and isosurface plotting (PyVista) |
| `openmodalpy[mkl]` | Intel MKL FFT backend |
| `openmodalpy[gpu]` | CuPy / PyTorch FFT backends |
| `openmodalpy[nek]` | Nek5000 field and mesh reader (pymech, GPL-3.0-or-later; see NOTICE) |

## Sixty seconds

Nothing to download. The built-in generators build a dataset in memory, and
the metadata they return carries the analytic answer (keys like
`expected_freq`, `St`, `f_shed`, `dmd_eigenvalue`, `decay_rate`), so you can
check the package against its own stated truth.

```python
from openmodalpy import PODAnalyzer, generate_double_gyre

data = generate_double_gyre(Nx=80, Ny=40, Nt=200)
pod = PODAnalyzer(data=data, n_modes_save=5)
pod.run_analysis(plots=False)

fractions = pod.eigenvalues / pod.total_energy
print(fractions)  # leading mode energy fractions
```

`run_analysis` writes the mode file into `results/` and, unless `plots=False`,
the figures into `figures/` under the current directory.

## One loader, every method

Load once, hand the same dictionary to each analyzer through `data=`, and
nothing re-reads the disk. Everything else in the package exists to make this
loop work.

```python
from openmodalpy import DMDAnalyzer, PODAnalyzer, SPODAnalyzer, generate_cylinder_wake

d = generate_cylinder_wake()          # or your own loader, see "Your data"

for cls, perform, kw in ((PODAnalyzer, "perform_pod", {}),
                         (SPODAnalyzer, "perform_spod", {}),
                         (DMDAnalyzer, "perform_dmd", {"rank": "svht"})):
    a = cls(data=d, **kw)
    a.load_and_preprocess()
    getattr(a, perform)()
    a.save_results()
```

Adding a fourth method is one more line in the tuple. DMD asks for a
truncation rank because there is no safe default. `"svht"` picks it from the
singular values. Every analyzer also takes `file_path=` for the shipped
readers and `run_analysis()` for the full load, decompose, save, plot
sequence.

`examples/compare_pod_spod.py` is the worked version. It runs POD and SPOD on
one dataset and plots them side by side. Its docstring explains what the
figure shows, and why the two leading POD modes come out as a near-equal pair
while SPOD gives one mode at one frequency.

## Methods

These are the names `openmodalpy methods list` reports and the values the
`method` field takes in a config file.

| `method` | Family | What it extracts | Reference |
|----------|--------|------------------|-----------|
| `pod` | energy | energy-ranked spatial modes | Lumley (1967); Sirovich (1987) |
| `mpod` | energy | modes separated by time-scale band | Mendez et al. (2019) |
| `psd-pod` | energy | POD of blockwise Fourier realizations | |
| `spod` | energy | modes at each frequency, from Welch blocks | Towne, Schmidt & Colonius (2018) |
| `stpod` | energy | space-time structures via delay embedding | |
| `dmd` | dynamics | modes with frequency and growth rate | Schmid (2010); Tu et al. (2014) |
| `hodmd` | dynamics | delay-embedded (Hankel) DMD | Le Clainche & Vega (2017) |
| `tls-hodmd` | dynamics | delay-embedded DMD, total-least-squares fit | Hemati et al. (2017) |
| `bsmd` | interactions | nonlinear triad structures | Schmidt (2020) |

`dmd` accepts `method: "ls"` (least squares) or `method: "tls"` (total least
squares, de-biased for noisy data). The TLS advantage on noisy data holds at
`embedding_dim=1` and decays as the embedding grows. By `embedding_dim=5`
plain LS is closer on average, so do not combine `tls` with a large embedding
to fight noise.

`hodmd` and `tls-hodmd` are `DMDAnalyzer` with a delay embedding. Call
`perform_dmd(embedding_dim=<d>, method="ls")` or `method="tls"`, where `<d>`
is the embedding depth.

The BSMD implementation follows Schmidt (2020) and was inspired by the
reference [MATLAB implementation](https://github.com/olivertschmidt/bmd).

`DOC.md` has one section per method with the equations, the output shapes
and the known limits of each implementation.

## Configuration and CLI

One JSONC file runs several methods over the same dataset:

```jsonc
{
  "case": {
    "name": "my_case",
    "data": { "kind": "file", "path": "data.mat" },
    "n_modes_save": 10, "nfft": 128, "overlap": 0.5,
    "rank": "svht"
  },
  "runs": [
    { "id": "pod",   "method": "pod" },
    { "id": "spod",  "method": "spod" },
    { "id": "dmd",   "method": "dmd",   "params": { "method": "ls" } },
    { "id": "hodmd", "method": "hodmd", "params": { "embedding_dim": 4 } },
    { "id": "bsmd",  "method": "bsmd" }
  ]
}
```

```bash
openmodalpy run --config analysis.jsonc            # full suite
openmodalpy run --config analysis.jsonc --dry-run  # preview without computing
openmodalpy analyze pod --config analysis.jsonc    # one method
openmodalpy methods list                           # supported methods
openmodalpy examples list                          # bundled examples
openmodalpy results inspect output.hdf5            # inspect a result file
```

## Bundled examples

Nine configs ship with the package, each running every method on one case. Three build their data from a closed-form field and run with nothing
to download. The others read a dataset from the path in their config, relative
to the config file.

| Example | Data | Case |
|---|---|---|
| `double_gyre` | generated | time-periodic double gyre, forcing frequency known |
| `cylinder_wake` | generated | von Karman cylinder wake, shedding Strouhal number known |
| `taylor_green` | generated | decaying Taylor-Green vortex, decay rate and DMD eigenvalue known |
| `cavity` | `.mat` file | experimental PIV of an open cavity |
| `jet` | `.mat` file | LES of a turbulent jet |
| `jet_small` | `.mat` file | reduced jet LES, for quick runs |
| `cylinder` | dNami NPZ directory | cylinder wake with spatial stride-2 loading |
| `cylinder_wake_compressible` | dNamiX NPZ | compressible cylinder wake |
| `run_benchmarks` | suite | runs the three generated cases in one go |

```bash
openmodalpy examples list
openmodalpy examples show double_gyre
openmodalpy examples run double_gyre
```

## Your data

The shipped readers auto-detect the file type:

- MATLAB `.mat`
- NumPy `.npz`, plain layout or the dNami family of consolidated and split layouts
- HDF5 `.h5` / `.hdf5`
- a directory of dNami split NPZ files
- Nek5000 `.f0*` field files, with the spectral-element quadrature weights, via the `nek` extra

Each must provide, or be readable as:

```python
{
    "q": np.ndarray,   # (Ns, Nspace)  snapshots x spatial points, required
    "dt": float,       # time step, required
    "x": np.ndarray,   # x-coordinates, required
    "y": np.ndarray,   # y-coordinates, required
    # "z", "t", "Nx", "Ny", "Nz", "Ns"   optional; derived when absent
}
```

Anything else is a plain function that returns that dictionary. Copy
`examples/my_data_template.py` for a commented starting point, or write it by
hand:

```python
def my_loader(path):
    return {"q": q, "dt": 0.01, "x": x, "y": y}

d = my_loader("run_001")        # one load
pod = PODAnalyzer(data=d)       # hand loaded data straight in
```

Spatial weights come with the data: uniform, polar, or cell volumes on a
stretched grid, and any prescribed weight vector you pass in. See `DOC.md`,
"Data Contract" and "Spatial weights".

## How we know it is right

The package is meant to be the one whose numbers you can defend in a review
response. That is a property of the tests, not of the prose.

- **Closed-form checks.** The tests run POD, SPOD, DMD and ST-POD on fields
  whose modes, energies, frequencies and growth rates are known analytically,
  and each test states its tolerance.
- **Cross-checks against other packages.** The tests compare SPOD eigenvalues
  with values PySPOD computed, and DMD eigenvalues with values PyDMD computed.
  The repository stores those reference numbers with the exact package
  versions that produced them. Neither package is a dependency.
- **Save and reload on every method.** One loader feeds every analyzer, each
  analyzer writes its result to HDF5 and reads it back, and the arrays must
  match bit for bit.
- **Mutation testing.** A monthly run alters the numerical core one line at a
  time and checks that a test fails for each change.
- **Speed tripwire.** A test fails when POD, SPOD or DMD gets far slower than
  the recorded time.
- **Provenance in every result.** Each HDF5 file carries the package
  versions, FFT backend, thread count, config hash, seed, git commit and
  timestamp that produced it.
- **CI on three platforms.** Linux, macOS and Windows, Python 3.11 to 3.14,
  with a coverage floor that only moves up.

Where an implementation is partial, unweighted, or uses a simplification, the
method section in `DOC.md` says so.

## Add your own method

A new decomposition is a subclass of `BaseAnalyzer` that implements one
`perform_<method>()` and fills `modes`, `eigenvalues` and `time_coefficients`.
The base handles loading, spatial weights, HDF5 save and reload, provenance,
mode-count truncation and the plotting hooks.

The cost is measured. `tests/toy_analyzer.py` is a deliberately trivial method
written against the base, and it is 137 lines including docstrings. The same
save-and-reload test that the shipped methods pass runs on it.

Reaching the CLI takes three more edits in shipped code: an entry in the
method registry, an import, and a dispatch line. `DOC.md`, "Adding an
Analyzer", walks through both the Python path and the CLI path and names what
is still clumsy about the second.

## A path through the examples

1. `openmodalpy examples run double_gyre`. One forcing frequency, so every
   method should find it. Check the SPOD peak against `expected_freq`.
2. Run `examples/compare_pod_spod.py`. Read its docstring first. It explains
   why POD splits a travelling structure into a sine and cosine pair and SPOD
   does not.
3. Run `taylor_green`. The field decays with no oscillation, so DMD should
   return one real eigenvalue equal to `dmd_eigenvalue` in the metadata.
4. Run `cylinder_wake` with `dmd` at `method: "ls"` and `method: "tls"`, then
   with `hodmd` at a few `embedding_dim` values, and watch the noise
   sensitivity described under Methods.
5. Open the matching section of `DOC.md` for the equations behind each step.

## FFT backend

FFT dispatch comes from [`fftkit`](https://github.com/openfluids/fftkit),
installed automatically. It picks the fastest backend present and falls back
to SciPy. To pin one:

```bash
export FFTKIT_BACKEND=mkl      # or scipy, numpy, cupy, accelerate
```

## Documentation

- [`DOC.md`](DOC.md): technical reference. Architecture, data contract, one
  section per method, config schema, CLI, output format, testing.
- [`CHANGELOG.md`](CHANGELOG.md): every release, with breaking changes to
  results noted as loudly as breaking changes to the API.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): setup and the checks CI runs.

## Contributing

Contributions are welcome, and questions and bug reports count. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).

## License

Apache-2.0. Originally developed by Ricardo A S Frantz. See [LICENSE](LICENSE)
and [NOTICE](NOTICE) for terms and attribution.
