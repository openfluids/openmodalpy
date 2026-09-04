#!/usr/bin/env python3
"""
Nek5000 spectral element data loader with Gauss-Lobatto-Legendre quadrature.

Reads binary Nek5000 field files and constructs quadrature weights for accurate
spatial integration on spectral element meshes. Weights account for the element
Jacobian and are ready for weighted modal decomposition.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import legendre

from openmodalpy.core.io import DataLoader, _infer_dt_from_times, natural_sort_key

logger = logging.getLogger(__name__)


def gll_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Lobatto-Legendre nodes and weights on [-1, 1].

    For n points in one direction, the nodes include the endpoints -1 and 1
    and interior points are the roots of the derivative of the Legendre
    polynomial P_{n-1}.

    Args:
        n (int): Number of points.

    Returns:
        tuple: (nodes, weights) both shape (n,). Nodes are in [-1, 1],
               weights sum to 2.
    """
    N = n - 1
    c = np.zeros(n)
    c[N] = 1.0  # Coefficients of P_N

    # Interior nodes: roots of P_N'
    interior = legendre.legroots(legendre.legder(c))
    interior = np.sort(interior)

    # Prepend and append boundary points
    nodes = np.concatenate(([-1.0], interior, [1.0]))

    # Weights: 2 / (N(N+1) P_N(x_i)^2)
    pn = legendre.legval(nodes, c)
    weights = 2.0 / (N * (N + 1) * pn**2)

    return nodes, weights


def differentiation_matrix(n: int, nodes: np.ndarray) -> np.ndarray:
    """
    Construct the Lagrange differentiation matrix at GLL nodes.

    The matrix D[i, j] represents d/dx evaluated at node i for the Lagrange
    basis function centered at node j.

    Args:
        n (int): Number of nodes.
        nodes (np.ndarray): The GLL nodes on [-1, 1].

    Returns:
        np.ndarray: Differentiation matrix, shape (n, n).
    """
    N = n - 1
    c = np.zeros(n)
    c[N] = 1.0  # P_N
    pn = legendre.legval(nodes, c)

    D = np.zeros((n, n))

    # Diagonal elements
    D[0, 0] = -N * (N + 1) / 4.0
    D[N, N] = N * (N + 1) / 4.0

    # Off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = pn[i] / (pn[j] * (nodes[i] - nodes[j]))

    return D


class NekDataLoader(DataLoader):
    """
    Loader for Nek5000 spectral element field files.

    Reads `.f0*NN` binary files and computes Gauss-Lobatto-Legendre quadrature
    weights on each element, accounting for the element Jacobian to provide
    accurate spatial integration weights.
    """

    def supports_format(self, file_path: str) -> bool:
        """Check if file_path contains or is a Nek5000 field file."""
        if file_path.lower().endswith(".f00001"):
            return True
        if os.path.isdir(file_path):
            return any(name.lower().endswith(".f00001") for name in os.listdir(file_path))
        return False

    def load(
        self,
        file_path: str,
        *,
        preview_ns: int | None = None,
        field: str | None = None,
        load_single: bool = False,
        schema: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """
        Load Nek5000 field files and return the standardized data contract.

        The loader reads one or more `.f0*NN` files and extracts a single
        variable (velocity component or pressure) at all snapshots and
        spatial points. Quadrature weights combine one-dimensional
        GLL weights and the element Jacobian determinant.

        Args:
            file_path (str): Path to a field file or directory containing them.
            preview_ns (int | None): Load only the first preview_ns snapshots.
            field (str | None): Variable and component to load, e.g. "u_1" for
                                first velocity component. Defaults to "u_1".
            load_single (bool): Ignored; accepted for interface compatibility.
            schema (dict | None): Ignored; accepted for interface compatibility.
            **kwargs: Unexpected keywords raise TypeError.

        Returns:
            dict[str, Any]: Data contract with keys:
                - q: (Ns, Nspace) array, the chosen variable at all points.
                - x, y[, z]: (Nspace,) coordinate arrays.
                - t: (Ns,) times from file headers.
                - dt: Inferred from times, or None if undeterminable.
                - spatial_weights: (Nspace,) product of GLL weights and |det J|.
                - spatial_weight_type: "prescribed" (these are non-uniform).
                - metadata: Variable name, component, element count, order.
        """
        del load_single, schema  # Unused
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Nek loader options: {unexpected}.")

        try:
            import pymech
        except ImportError as e:
            raise ImportError("Reading Nek5000 files needs the nek extra: pip install openmodalpy[nek]") from e

        # Resolve file list
        file_paths = self._resolve_field_files(file_path)
        logger.info("Loading Nek5000 field files (count: %d)", len(file_paths))

        if not file_paths:
            raise FileNotFoundError(f"No Nek5000 field files found at {file_path}")

        # Read first file to get mesh metadata
        data_first = pymech.readnek(file_paths[0])
        ndim = data_first.ndim
        nel = data_first.nel
        lr1 = data_first.lr1  # (nx1, ny1, nz1)
        nx1, ny1, nz1 = lr1

        # Parse field argument
        var_name, comp = self._parse_field(field)
        comp_idx = comp - 1  # User names components 1–3; array index is 0–2

        # Read all files and collect snapshots
        all_times: list[float] = []
        all_snapshots_list: list[np.ndarray] = []

        for file_idx, fpath in enumerate(file_paths):
            data = pymech.readnek(fpath)
            if data.ndim != ndim or data.nel != nel or data.lr1 != lr1:
                raise ValueError(f"File {fpath} has inconsistent mesh or order (expected nel={nel}, lr1={lr1})")

            # Extract times for this file
            if data.time:
                all_times.extend(data.time)
            else:
                all_times.extend([float(file_idx)])

            # Extract field at this time across all elements
            snapshot = np.zeros((nel, nz1, ny1, nx1))
            for elem_idx, elem in enumerate(data.elem):
                if var_name == "u":
                    snapshot[elem_idx] = elem.vel[comp_idx]
                elif var_name == "p":
                    snapshot[elem_idx] = elem.pres[0]
                else:
                    snapshot[elem_idx] = elem.scal[0]

            all_snapshots_list.append(snapshot)

        # Stack snapshots: each element of all_snapshots_list has shape (nel, nz1, ny1, nx1)
        # We want to stack them along a new time dimension: (Ns, nel, nz1, ny1, nx1)
        snapshots_stacked = np.stack(all_snapshots_list, axis=0)  # (Ns, nel, nz1, ny1, nx1)

        if preview_ns is not None:
            snapshots_stacked = snapshots_stacked[:preview_ns]
            all_times = all_times[:preview_ns]

        Ns = snapshots_stacked.shape[0]

        # Get data from first file for mesh and compute weights once
        data = pymech.readnek(file_paths[0])

        # Compute GLL nodes and weights for this order
        nodes_r, w_r = gll_nodes_and_weights(nx1)
        nodes_s, w_s = gll_nodes_and_weights(ny1)
        if ndim == 3:
            nodes_t, w_t = gll_nodes_and_weights(nz1)

        # Differentiation matrices for Jacobian
        D_r = differentiation_matrix(nx1, nodes_r)
        D_s = differentiation_matrix(ny1, nodes_s)
        if ndim == 3:
            D_t = differentiation_matrix(nz1, nodes_t)

        # Compute spatial coordinates and weights from first file
        all_x = []
        all_y = []
        all_z: list[np.ndarray] = []
        all_weights = []

        for elem_idx, elem in enumerate(data.elem):
            pos = elem.pos  # (3, nz1, ny1, nx1): x, y, z at each node
            x_phys = pos[0]  # (nz1, ny1, nx1)
            y_phys = pos[1]
            z_phys = pos[2]

            # Compute Jacobian for this element
            if ndim == 2:
                # 2D: det J = (dx/dr)(dy/ds) - (dx/ds)(dy/dr)
                # Extract 2D slice and operate on it
                x_2d = x_phys[0]  # (ny1, nx1)
                y_2d = y_phys[0]
                dx_dr = np.dot(D_r, x_2d.T).T  # (ny1, nx1)
                dx_ds = np.dot(D_s, x_2d)  # (ny1, nx1)
                dy_dr = np.dot(D_r, y_2d.T).T
                dy_ds = np.dot(D_s, y_2d)

                # Compute det J at each point
                detJ = dx_dr * dy_ds - dx_ds * dy_dr
            else:  # ndim == 3
                # 3D: full 3x3 Jacobian determinant
                # r-direction (last axis, length nx1)
                # s-direction (middle axis, length ny1)
                # t-direction (first axis, length nz1)
                dx_dr = np.zeros((nz1, ny1, nx1))
                dx_ds = np.zeros((nz1, ny1, nx1))
                dx_dt = np.zeros((nz1, ny1, nx1))
                dy_dr = np.zeros((nz1, ny1, nx1))
                dy_ds = np.zeros((nz1, ny1, nx1))
                dy_dt = np.zeros((nz1, ny1, nx1))
                dz_dr = np.zeros((nz1, ny1, nx1))
                dz_ds = np.zeros((nz1, ny1, nx1))
                dz_dt = np.zeros((nz1, ny1, nx1))

                for i in range(nz1):
                    for j in range(ny1):
                        dx_dr[i, j, :] = np.dot(D_r, x_phys[i, j, :])
                        dy_dr[i, j, :] = np.dot(D_r, y_phys[i, j, :])
                        dz_dr[i, j, :] = np.dot(D_r, z_phys[i, j, :])

                for i in range(nz1):
                    for k in range(nx1):
                        dx_ds[i, :, k] = np.dot(D_s, x_phys[i, :, k])
                        dy_ds[i, :, k] = np.dot(D_s, y_phys[i, :, k])
                        dz_ds[i, :, k] = np.dot(D_s, z_phys[i, :, k])

                for j in range(ny1):
                    for k in range(nx1):
                        dx_dt[:, j, k] = np.dot(D_t, x_phys[:, j, k])
                        dy_dt[:, j, k] = np.dot(D_t, y_phys[:, j, k])
                        dz_dt[:, j, k] = np.dot(D_t, z_phys[:, j, k])

                # Compute 3x3 determinant
                detJ = (
                    dx_dr * (dy_ds * dz_dt - dy_dt * dz_ds)
                    - dx_ds * (dy_dr * dz_dt - dy_dt * dz_dr)
                    + dx_dt * (dy_dr * dz_ds - dy_ds * dz_dr)
                )

            # Weight: outer product of 1D weights times |det J|
            if ndim == 2:
                w_2d = np.outer(w_s, w_r)  # w_s on rows, w_r on columns
                elem_weights = w_2d * np.abs(detJ)
            else:  # ndim == 3
                w_3d = np.zeros((nz1, ny1, nx1))
                for i in range(nz1):
                    w_3d[i] = np.outer(w_s, w_r)
                w_3d = w_t[:, None, None] * w_3d
                elem_weights = w_3d * np.abs(detJ)

            # Flatten and collect
            all_x.append(np.ravel(x_phys, order="C"))
            all_y.append(np.ravel(y_phys, order="C"))
            if ndim == 3:
                all_z.append(np.ravel(z_phys, order="C"))
            all_weights.append(np.ravel(elem_weights, order="C"))

        # Concatenate over all elements to get spatial grid
        x = np.concatenate(all_x)  # (Nspace,)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z) if ndim == 3 else None
        weights = np.concatenate(all_weights)
        Nspace = x.shape[0]

        # Reshape snapshots: (Ns, nel, nz1, ny1, nx1) -> (Ns, Nspace)
        q = snapshots_stacked.reshape(Ns, Nspace)

        # Times
        times = np.array(all_times)
        dt = _infer_dt_from_times(times)

        logger.info(
            "Loaded Nek5000 data: Ns=%d, Nspace=%d, ndim=%d, "
            "nel=%d, order=(nx1=%d, ny1=%d, nz1=%d), "
            "var=%s, component=%d, dt=%s",
            Ns,
            Nspace,
            ndim,
            nel,
            nx1,
            ny1,
            nz1,
            var_name,
            comp,
            dt,
        )

        return {
            "q": q,
            "x": x,
            "y": y,
            "z": z,
            "t": times,
            "dt": dt,
            "spatial_weights": weights,
            "spatial_weight_type": "prescribed",
            "metadata": {
                "format": "nek",
                "file_path": file_path,
                "var_name": var_name,
                "component": comp,
                "nel": nel,
                "lr1": lr1,
                "ndim": ndim,
            },
        }

    @staticmethod
    def _resolve_field_files(file_path: str) -> list[str]:
        """
        Resolve a directory or glob pattern to a list of Nek5000 field files.

        Nek5000 files follow the pattern `case.f0*NN`, sorted naturally.

        Args:
            file_path (str): Directory or glob pattern.

        Returns:
            list[str]: Sorted absolute paths to field files.
        """
        root = Path(file_path).expanduser()
        if root.is_dir():
            pattern = str(root / "*.f[0-9][0-9][0-9][0-9][0-9]")
        else:
            pattern = str(root)

        matches = glob.glob(pattern)
        if not matches:
            # Try alternate pattern
            alt_pattern = str(root / "*.f0*") if root.is_dir() else str(root).replace(".f00001", ".f*")
            matches = glob.glob(alt_pattern)

        return sorted(set(matches), key=natural_sort_key)

    @staticmethod
    def _parse_field(field: str | None) -> tuple[str, int]:
        """
        Parse field specification into variable name and component.

        Args:
            field (str | None): Field spec like "u_1", "p", etc.
                                Defaults to "u_1".

        Returns:
            tuple[str, int]: (variable_name, component_number). Component is 1–3
                             for velocity, 1 for pressure.
        """
        if field is None:
            field = "u_1"

        if "_" in field:
            parts = field.split("_")
            var_name = parts[0]
            try:
                comp = int(parts[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid field specification: {field}. Use format like 'u_1', 'u_2', 'p'.") from None
        else:
            var_name = field
            comp = 1

        if var_name not in ("u", "p"):
            raise ValueError(f"Unsupported variable '{var_name}'. Use 'u' or 'p'.")

        if var_name == "u" and comp not in (1, 2, 3):
            raise ValueError(f"Invalid velocity component {comp}. Use 1, 2, or 3.")
        if var_name == "p" and comp != 1:
            raise ValueError(f"Pressure has only one component; requested {comp}.")

        return var_name, comp
