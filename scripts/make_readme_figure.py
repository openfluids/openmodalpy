#!/usr/bin/env python3
"""Draw the README figure from the shipped cylinder-wake generator.

Developer-only. Never collected or run by pytest. The figure shows what the
package produces on a case whose shedding frequency is known in closed form,
so the reader sees the answer and the check in one picture.

    uv run python scripts/make_readme_figure.py

Writes assets/readme_cylinder_wake_pod_spod.png at 400 dpi and the measured
numbers beside it in assets/readme_cylinder_wake_pod_spod.log. The generator
carries a fixed seed, so a re-run reproduces both.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from openmodalpy import PODAnalyzer, SPODAnalyzer, generate_cylinder_wake

STEM = "readme_cylinder_wake_pod_spod"
# nfft 100 puts a Welch bin within 0.2 % of the closed-form shedding Strouhal
# and still leaves 9 blocks to average. A block that does not divide the record
# into a whole number of shedding periods moves the peak a whole bin.
NFFT = 100
OVERLAP = 0.5
N_MODES = 6


def main() -> None:
    """Run POD and SPOD on the cylinder wake and draw the three panels."""
    data = generate_cylinder_wake()
    nx, ny = int(data["Nx"]), int(data["Ny"])
    st_shed = float(data["metadata"]["St"])

    pod = PODAnalyzer(data=data, spatial_weight_type="uniform", n_modes_save=N_MODES)
    pod.load_and_preprocess()
    pod.perform_pod()
    energy_fraction = np.asarray(pod.eigenvalues) / float(pod.total_energy)

    spod = SPODAnalyzer(data=data, spatial_weight_type="uniform", nfft=NFFT, overlap=OVERLAP, n_modes_save=2)
    spod.load_and_preprocess()
    spod.perform_spod()
    strouhal = np.asarray(spod.St)
    spod_energy = np.asarray(spod.eigenvalues)
    peak_st = float(strouhal[int(np.argmax(spod_energy[:, 0]))])

    mode = np.asarray(pod.modes)[:, 0].real.reshape(ny, nx)
    limit = float(np.max(np.abs(mode)))
    x_mesh, y_mesh = np.meshgrid(np.asarray(data["x"]), np.asarray(data["y"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    ax = axes[0]
    ax.contourf(x_mesh, y_mesh, mode, levels=np.linspace(-limit, limit, 21), cmap="RdBu_r")
    ax.set_title("(a) leading POD mode", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    ax = axes[1]
    ranks = np.arange(1, energy_fraction.size + 1)
    ax.plot(ranks, energy_fraction, marker="o", linewidth=1.4, color="#1f4e79", label="POD mode energy")
    ax.set_yscale("log")
    ax.set_title("(b) energy per mode", fontsize=10)
    ax.set_xlabel("mode number")
    ax.set_ylabel("fraction of total energy")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    # Thickest curve at the back: the second mode is drawn first.
    ax.plot(
        strouhal,
        spod_energy[:, 1],
        marker="s",
        markersize=3,
        markevery=3,
        linewidth=2.6,
        color="#9ecae1",
        label="SPOD mode 2",
    )
    ax.plot(
        strouhal,
        spod_energy[:, 0],
        marker="o",
        markersize=3,
        markevery=3,
        linewidth=1.3,
        color="#1f4e79",
        label="SPOD mode 1",
    )
    ax.axvline(st_shed, linewidth=1.0, linestyle="--", color="0.35", label=f"closed form St = {st_shed:.3f}")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("(c) SPOD spectrum", fontsize=10)
    ax.set_xlabel("Strouhal number")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "assets"
    figure_path = out / f"{STEM}.png"
    fig.savefig(figure_path, dpi=400)
    plt.close(fig)

    log = [
        f"case: cylinder_wake Nx={nx} Ny={ny} Ns={int(data['Ns'])} dt={float(data['dt']):.6f} s",
        f"closed-form shedding Strouhal: {st_shed:.6f}",
        f"SPOD peak Strouhal (mode 1): {peak_st:.6f}",
        f"POD energy fractions: {np.array2string(energy_fraction, precision=4)}",
        f"SPOD blocks: {int(spod_energy.shape[1])}, nfft={NFFT}, overlap={OVERLAP}",
    ]
    (out / f"{STEM}.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
