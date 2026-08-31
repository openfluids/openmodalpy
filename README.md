![openmodalpy banner](https://raw.githubusercontent.com/openfluids/openmodalpy/main/assets/readme-banner-v3.jpg)

[![CI](https://github.com/openfluids/openmodalpy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/openfluids/openmodalpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openmodalpy.svg)](https://pypi.org/project/openmodalpy/)
[![Python](https://img.shields.io/pypi/pyversions/openmodalpy.svg)](https://pypi.org/project/openmodalpy/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`openmodalpy` puts nine modal decomposition methods for spatiotemporal data
behind one API. Extract coherent structures from simulation or experimental
data — energy-ranked POD modes, frequency-resolved SPOD modes, DMD eigenvalues,
nonlinear BSMD triads — without switching libraries or rewriting your loading
code for each method.

## Why this package

Most Python tools here specialise: [PyDMD](https://github.com/PyDMD/PyDMD) covers DMD
variants in depth, [PySPOD](https://github.com/MathEXLab/PySPOD) covers SPOD. That depth
is real, and if you only need one method they are excellent choices.

OpenModalPy trades some of that depth for breadth. Nine methods share one analyzer
interface, one data contract and one config file, so running POD, SPOD, DMD and BSMD over
the same dataset — and comparing them directly — is a single command rather than four
integrations. Bispectral mode decomposition (BSMD) in particular has little open-source
coverage elsewhere.

It runs on the NumPy/SciPy stack. No compiled solver toolchain (PETSc, SLEPc) to install.

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

## Quick Start

Nothing to download: the built-in generators produce a synthetic dataset in memory, and
their returned metadata carries the analytic ground truth (keys like `expected_freq`, `St`,
`f_shed`, `dmd_eigenvalue`, `decay_rate`), so you can check the package's output against its
own stated truth.

```python
from openmodalpy import PODAnalyzer, generate_double_gyre

data = generate_double_gyre(Nx=80, Ny=40, Nt=200)
pod = PODAnalyzer(data=data, n_modes_save=5)
pod.run_analysis(plots=False)

fractions = pod.eigenvalues / pod.total_energy
print(fractions)  # leading mode energy fractions
```

`run_analysis` writes the mode file into `results/` (and, unless `plots=False`, the figures
into `figures/`) under the current directory.

With your own data on disk:

```python
from openmodalpy import PODAnalyzer, SPODAnalyzer, DMDAnalyzer

pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
pod.run_analysis()

spod = SPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
spod.run_analysis()

# DMD is driven in two steps, so the fit method can be chosen after loading.
dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10)
dmd.load_and_preprocess()
dmd.perform_dmd(method="ls")
```

## Configuration-Driven Workflow

One JSONC file runs several methods over the same dataset — the main reason to reach for
this package over a single-method library:

```jsonc
{
  "case": {
    "name": "my_case",
    "data": { "kind": "file", "path": "data.mat" },
    "n_modes_save": 10, "nfft": 128, "overlap": 0.5
  },
  "runs": [
    { "id": "pod",   "method": "pod" },
    { "id": "spod",  "method": "spod" },
    { "id": "dmd",   "method": "dmd",   "params": { "method": "ls" } },
    { "id": "hodmd", "method": "hodmd", "params": { "delays": 4 } },
    { "id": "bsmd",  "method": "bsmd" }
  ]
}
```

```bash
openmodalpy run --config analysis.jsonc
```

## CLI

```bash
openmodalpy analyze pod --config case.jsonc     # one method
openmodalpy run --config suite.jsonc            # full suite
openmodalpy run --config suite.jsonc --dry-run  # preview without computing
openmodalpy methods list                        # supported methods
openmodalpy examples list                       # bundled examples
openmodalpy results inspect output.hdf5         # inspect a result file
```

Three example cases ship with the package and need no external data — `double_gyre`,
`cylinder_wake` and `taylor_green` generate their fields analytically. A fourth config,
`run_benchmarks`, runs all three as a suite. So `openmodalpy examples list` gives you
something runnable immediately, with nothing to download.

## Methods

These are the names `openmodalpy methods list` reports and the values the `method` field
takes in a config file.

| `method` | Class | What it extracts | Reference |
|----------|-------|------------------|-----------|
| `pod` | variance-optimal | energy-ranked spatial modes | Lumley (1967); Sirovich (1987) |
| `mpod` | variance-optimal | scale-separated modes across non-overlapping bands | [Mendez et al. (2019)](https://doi.org/10.1017/jfm.2019.212) |
| `psd-pod` | variance-optimal | POD of blockwise Fourier realizations | — |
| `spod` | variance-optimal | frequency-local modes (Welch blocks) | [Towne, Schmidt & Colonius (2018)](https://doi.org/10.1017/jfm.2018.283) |
| `stpod` | variance-optimal | space-time structures via delay embedding | — |
| `dmd` | evolution-fit | modes with frequency and growth rate | [Schmid (2010)](https://doi.org/10.1017/S0022112010001217); [Tu et al. (2014)](https://doi.org/10.3934/jcd.2014.1.391) |
| `hodmd` | evolution-fit | delay-embedded (Hankel) DMD | [Le Clainche & Vega (2017)](https://doi.org/10.1137/15M1054924) |
| `tls-hodmd` | evolution-fit | delay-embedded DMD, total-least-squares fit | [Hemati et al. (2017)](https://doi.org/10.1007/s00162-017-0432-2) |
| `bsmd` | triadic interaction | nonlinear triad structures | [Schmidt (2020)](https://doi.org/10.1007/s11071-020-06037-z) |

`dmd` accepts `method: "ls"` (least squares) or `method: "tls"` (total least squares,
de-biased for noisy data).

The BSMD implementation follows Schmidt (2020) and was inspired by the reference
[MATLAB implementation](https://github.com/olivertschmidt/bmd).

To compare two methods on one dataset through the Python API, see
`examples/compare_pod_spod.py`: it loads one dataset once, runs POD and SPOD
on it, and plots both.

## Data Format

`.mat` and `.npz` files are auto-detected and must provide:

```python
{
    "q": np.ndarray,   # (Ns, Nspace) — snapshots × spatial points, required
    "dt": float,       # time step, required
    "x": np.ndarray,   # x-coordinates, required
    "y": np.ndarray,   # y-coordinates, required
    # "Nx": int,       # grid points in x — derived from q, x, y when absent
    # "Ny": int,       # grid points in y — derived from q, x, y when absent
}
```

Anything else can be read with a custom loader returning the same dictionary.
Copy `examples/my_data_template.py` for a fully commented starting point, or
write the dict by hand:

```python
def my_loader(path):
    return {"q": data, "dt": 0.01, "x": x, "y": y}

d = my_loader("run_001")        # one load — or build the dict yourself

pod = PODAnalyzer(data=d)       # hand loaded data straight in; no fake path
```

See `DOC.md`, "Your own format", for the full contract and the plug-in point
this uses.

## FFT Backend

FFT dispatch comes from [`fftkit`](https://github.com/openfluids/fftkit), installed
automatically. It probes the available backends, picks the fastest, and falls back to
SciPy when nothing else is present — so this section is optional reading.

To pin a backend:

```bash
export FFTKIT_BACKEND=mkl      # or scipy, numpy, cupy, accelerate
```

```python
from openmodalpy.core import FFT_BACKEND
print(FFT_BACKEND)   # the backend actually in use
```

The legacy `PYMODAL_FFT_BACKEND` variable still works as a fallback, but
`FFTKIT_BACKEND` is the supported name.

## Contributing

Contributions are welcome, and questions and bug reports count. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup and the checks CI runs, and the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md)
for how we work together.

## License

Apache-2.0. Originally developed by Ricardo A S Frantz — see [LICENSE](LICENSE) and
[NOTICE](NOTICE) for terms and attribution.
