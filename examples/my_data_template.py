#!/usr/bin/env python3
"""Template: load your own data into the openmodalpy data contract.

Copy this file, rename it, and edit ``load_my_data`` for your dataset. The
returned dict is the one input every analyzer accepts through ``data=``.

Required keys: ``q``, ``x``, ``y``, ``dt``.
Derived when absent: ``Nx``, ``Ny``, ``Nz``, ``Ns`` (computed from the shapes
of ``q``, ``x``, ``y`` and ``z``, the same rule the built-in file readers use).

``q`` shape convention: ``(Ns, Nspace)``, snapshots stacked along axis 0,
one flattened spatial field per row. For a 2-D grid, flatten in C order:
``index = iy * Nx + ix``. For a 3-D grid, add ``iz * Ny * Nx``. ``x`` and
``y`` (and ``z`` for 3-D) are the coordinate arrays that grid came from.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_my_data(path: str) -> dict[str, Any]:
    """Load a dataset from ``path`` into the openmodalpy data contract.

    Replace the body with your own reader (NumPy save file, CSV directory,
    a simulation restart file, ...). This example builds a tiny synthetic
    field so the template runs on its own, with no external file needed.

    Parameters
    ----------
    path : str
        Location of the dataset on disk. Unused by the synthetic example;
        a real reader opens this path.

    Returns
    -------
    dict[str, Any]
        A dict following the openmodalpy data contract.
    """
    del path  # the synthetic example below ignores it; a real loader reads it

    # --- Replace this block with your own array-building code -------------
    n_snapshots = 12
    n_x, n_y = 5, 4
    dt_seconds = 0.1  # time step [s]
    x = np.linspace(0.0, 1.0, n_x)  # x-coordinates [m]
    y = np.linspace(0.0, 1.0, n_y)  # y-coordinates [m]
    time = np.arange(n_snapshots) * dt_seconds
    xx, yy = np.meshgrid(x, y)  # shape (n_y, n_x), matches C-order flattening
    q = np.stack(
        [np.sin(2.0 * np.pi * t + xx) * np.cos(yy) for t in time],
        axis=0,
    ).reshape(n_snapshots, n_x * n_y)
    # -------------------------------------------------------------------

    return {
        "q": q,  # (Ns, Nspace), snapshots x flattened space, required
        "x": x,  # x-coordinates [m], required
        "y": y,  # y-coordinates [m], required
        "dt": dt_seconds,  # time step [s], required
        # "z": z_coords,           # z-coordinates [m], only for 3-D data
        # "Nx": n_x,               # grid points in x, derived when absent
        # "Ny": n_y,               # grid points in y, derived when absent
        # "Nz": 1,                 # grid points in z, derived when absent
        # "Ns": n_snapshots,       # number of snapshots, derived when absent
    }


if __name__ == "__main__":
    from openmodalpy import PODAnalyzer

    data = load_my_data("path/to/your/dataset")
    pod = PODAnalyzer(data=data, n_modes_save=2)
    pod.load_and_preprocess()
    pod.perform_pod()
    print(f"POD ran on {pod.data['Ns']} snapshots, {pod.data['Nx']}x{pod.data['Ny']} grid.")
