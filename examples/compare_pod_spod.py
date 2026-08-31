#!/usr/bin/env python3
"""Compare two decompositions of the same dataset through the Python API.

This is the worked example for the package's core promise: write one loader,
then run every method you need on the same data. It builds one dataset in
memory, from the shipped cylinder-wake generator, and hands that single dict
to two analyzer classes through ``data=``. Neither analyzer re-reads a file.

POD and SPOD answer different questions about the same field:

- POD (:class:`~openmodalpy.PODAnalyzer`) ranks modes by total energy,
  pooled over the whole time series. It answers "what spatial pattern
  carries the most variance?"
- SPOD (:class:`~openmodalpy.SPODAnalyzer`) ranks modes by energy at each
  frequency separately. It answers "what pattern carries the most variance
  at this frequency?"

What to look for in the figure. The SPOD curve has one sharp peak, at the
vortex shedding frequency of 0.167 Hz. Its eigenvalue there is about 2600
times the value at zero frequency, so the wake is a narrowband oscillation,
not a broadband one. POD cannot report that: its curve is an energy ranking
with no frequency axis at all.

Look also at the first two POD modes. They carry 0.53 and 0.47 of the energy,
99.8 % together, and everything after them drops to 3e-4. That near-equal pair
is not a coincidence: a structure that travels cannot be written as one real
mode, so POD splits it into a sine and cosine pair of almost the same energy.
SPOD writes the same structure as ONE mode at ONE frequency. That is the
clearest difference between the two methods on this flow.

The two leading spatial modes look almost the same here, and that is the
correct answer for this flow, not a fault. They correlate at 0.96, because
shedding carries most of the energy, so the mode POD ranks first IS the
shedding mode. The gain from SPOD on this dataset is the frequency, not a
different shape. On a flow with two mechanisms at different frequencies the
shapes separate as well.

A note on the block length. SPOD resolves a frequency only when the block is
long enough. The generator samples at 8.34 Hz over 500 snapshots, so
``nfft=100`` gives a bin width of 0.083 Hz and puts the 0.167 Hz tone exactly
on a bin, with 9 Welch blocks to average. A shorter block hides the tone in
the zero-frequency bin, and a longer one leaves too few blocks to average.

Run directly, or import and call :func:`main` with an output directory (used
by the test that exercises this script).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openmodalpy import PODAnalyzer, SPODAnalyzer, generate_cylinder_wake


def main(output_dir: str | Path = ".") -> tuple[PODAnalyzer, SPODAnalyzer]:
    """Build one dataset, run POD and SPOD on it, and plot both.

    Parameters
    ----------
    output_dir : str | Path
        Directory for results (``<output_dir>/results``) and the
        comparison figure (``<output_dir>/figures``). Defaults to the
        current directory.

    Returns
    -------
    tuple[PODAnalyzer, SPODAnalyzer]
        The two analyzers, already run, so a caller can inspect them.
    """
    out = Path(output_dir)
    results_dir = str(out / "results")
    figures_dir = str(out / "figures")

    # One loader feeds both analyzers: build the dataset once, pass it
    # through data= to each constructor, load it nowhere else.
    dataset = generate_cylinder_wake()

    pod = PODAnalyzer(data=dataset, n_modes_save=10, results_dir=results_dir, figures_dir=figures_dir)
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results()

    # The wake sheds at 0.167 Hz and the generator samples at 8.34 Hz over 500
    # snapshots. nfft=100 gives a bin width of 0.083 Hz, which puts the shedding
    # tone on bin 2 exactly, and leaves 9 Welch blocks to average over.
    spod = SPODAnalyzer(data=dataset, nfft=100, results_dir=results_dir, figures_dir=figures_dir)
    spod.load_and_preprocess()
    spod.perform_spod()
    spod.save_results()

    figure_path = out / "figures" / "compare_pod_spod.png"
    _plot_comparison(pod, spod, figure_path)
    return pod, spod


def _plot_comparison(pod: PODAnalyzer, spod: SPODAnalyzer, figure_path: Path) -> None:
    """Write the four-panel comparison figure to ``figure_path``."""
    Nx, Ny = pod.data["Nx"], pod.data["Ny"]

    pod_energy_fraction = pod.eigenvalues / pod.total_energy
    leading_pod_mode = pod.modes[:, 0].reshape(Ny, Nx)

    spod_leading_eigenvalue = spod.eigenvalues[:, 0]
    peak_freq_idx = int(np.argmax(spod_leading_eigenvalue))
    leading_spod_mode = np.real(spod.modes[peak_freq_idx, :, 0]).reshape(Ny, Nx)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    ax = axes[0, 0]
    ax.semilogy(np.arange(1, pod_energy_fraction.size + 1), pod_energy_fraction, marker="o")
    ax.set_xlabel("POD mode index [-]")
    ax.set_ylabel("Energy fraction [-]")
    ax.set_title("POD: energy by mode")

    ax = axes[0, 1]
    ax.semilogy(spod.freq, spod_leading_eigenvalue, marker="o")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Leading SPOD eigenvalue [-]")
    ax.set_title("SPOD: energy by frequency")

    ax = axes[1, 0]
    im = ax.pcolormesh(pod.data["x"], pod.data["y"], leading_pod_mode, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Leading POD mode")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.pcolormesh(pod.data["x"], pod.data["y"], leading_spod_mode, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Leading SPOD mode at f={spod.freq[peak_freq_idx]:.3f} Hz")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
