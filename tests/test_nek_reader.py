#!/usr/bin/env python3
"""
Tests for Nek5000 field reader with quadrature weight computation.

Tests verify that GLL weights combined with element Jacobians produce
spectrally accurate integration over spectral element meshes, and that
unequal element sizes are handled correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openmodalpy.core.io import load_data


def test_nek_loader_equal_elements(tmp_path: Path) -> None:
    """
    Test Nek loader with equal-sized elements.

    Assert that the integral of sin(x)^2 cos(y)^2 over [0, pi]^2 matches
    the expected value pi^2/4 to high relative accuracy.
    """
    try:
        import pymech  # noqa: F401
    except ImportError:
        pytest.skip("pymech not installed")

    from openmodalpy.core.nek import gll_nodes_and_weights

    test_dir = tmp_path / "nek_test"
    test_dir.mkdir()

    # Create a dummy .f00001 file so the loader can find something
    dummy_file = test_dir / "case.f00001"
    dummy_file.touch()

    nx1, ny1, nel_x, nel_y = 5, 5, 2, 2
    nodes_gll, _ = gll_nodes_and_weights(nx1)
    nel = nel_x * nel_y
    nz1 = 1
    ndim = 2

    # Build test data
    elem_list = []
    for iy in range(nel_y):
        y_start = 0.0 + iy * (np.pi / nel_y)
        y_end = y_start + (np.pi / nel_y)

        for ix in range(nel_x):
            x_start = 0.0 + ix * (np.pi / nel_x)
            x_end = x_start + (np.pi / nel_x)

            x_scale = (x_end - x_start) / 2.0
            y_scale = (y_end - y_start) / 2.0

            x_phys = np.zeros((nz1, ny1, nx1))
            y_phys = np.zeros((nz1, ny1, nx1))

            for j in range(ny1):
                for i in range(nx1):
                    x_phys[0, j, i] = x_start + x_scale * (nodes_gll[i] + 1.0)
                    y_phys[0, j, i] = y_start + y_scale * (nodes_gll[j] + 1.0)

            vel = np.zeros((3, nz1, ny1, nx1))
            vel[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            pres = np.zeros((1, nz1, ny1, nx1))
            pres[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            elem_data = MagicMock()
            elem_data.pos = np.array([x_phys, y_phys, np.zeros_like(x_phys)])
            elem_data.vel = vel
            elem_data.pres = pres
            elem_data.scal = vel[:1]

            elem_list.append(elem_data)

    hexa_data = MagicMock()
    hexa_data.ndim = ndim
    hexa_data.nel = nel
    hexa_data.lr1 = (nx1, ny1, nz1)
    hexa_data.var = ["u", "v", "w", "p", "t"]
    hexa_data.time = [0.0]
    hexa_data.istep = [1]
    hexa_data.elem = elem_list

    with patch("pymech.readnek", return_value=hexa_data):
        # Load using interface
        data = load_data(str(test_dir), loader_type="nek", field="u_1")

        # Verify shapes
        q = data["q"]
        x = data["x"]
        y = data["y"]
        weights = data["spatial_weights"]

        assert q.ndim == 2
        assert q.shape[0] == 1  # One snapshot
        Nspace = q.shape[1]
        assert x.shape == (Nspace,)
        assert y.shape == (Nspace,)
        assert weights.shape == (Nspace,)

        # Compute integral: sum(w * f^2)
        f_values = q[0]  # sin(x) * cos(y)
        integral = np.sum(weights * f_values**2)
        expected = (np.pi / 2.0) ** 2  # (pi/2) * (pi/2)

        rel_error = np.abs(integral - expected) / expected
        print(f"Equal elements: integral={integral:.15e}, expected={expected:.15e}, rel_error={rel_error:.2e}")

        assert rel_error < 1e-10, f"Integral {integral} does not match expected {expected}; relative error {rel_error}"

        # Verify coordinates are consistent
        assert np.all(x >= 0.0) and np.all(x <= np.pi)
        assert np.all(y >= 0.0) and np.all(y <= np.pi)


def test_nek_loader_unequal_elements(tmp_path: Path) -> None:
    """
    Test Nek loader with unequal-sized elements in x.

    The same integral should hold; this test catches errors in Jacobian
    computation or reuse across elements.
    """
    try:
        import pymech  # noqa: F401
    except ImportError:
        pytest.skip("pymech not installed")

    from openmodalpy.core.nek import gll_nodes_and_weights

    test_dir = tmp_path / "nek_test_unequal"
    test_dir.mkdir()
    dummy_file = test_dir / "case.f00001"
    dummy_file.touch()

    # Nine points a side. GLL quadrature is exact only for polynomials, and
    # sin(x)^2 is not one, so the order sets the floor: on this mesh five points
    # give 2.0e-05 and nine give 5.2e-14. Nine puts the maths, not the
    # resolution, on trial.
    nx1, ny1, nel_x, nel_y = 9, 9, 2, 2
    nodes_gll, _ = gll_nodes_and_weights(nx1)
    nel = nel_x * nel_y
    nz1 = 1
    ndim = 2

    # Unequal element widths: first element 2*pi/3, second pi/3
    element_widths_x = [2.0 * np.pi / 3.0, np.pi / 3.0]

    elem_list = []
    for iy in range(nel_y):
        y_start = 0.0 + iy * (np.pi / nel_y)
        y_end = y_start + (np.pi / nel_y)

        for ix in range(nel_x):
            x_start = sum(element_widths_x[:ix])
            x_end = x_start + element_widths_x[ix]

            x_scale = (x_end - x_start) / 2.0
            y_scale = (y_end - y_start) / 2.0

            x_phys = np.zeros((nz1, ny1, nx1))
            y_phys = np.zeros((nz1, ny1, nx1))

            for j in range(ny1):
                for i in range(nx1):
                    x_phys[0, j, i] = x_start + x_scale * (nodes_gll[i] + 1.0)
                    y_phys[0, j, i] = y_start + y_scale * (nodes_gll[j] + 1.0)

            vel = np.zeros((3, nz1, ny1, nx1))
            vel[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            pres = np.zeros((1, nz1, ny1, nx1))
            pres[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            elem_data = MagicMock()
            elem_data.pos = np.array([x_phys, y_phys, np.zeros_like(x_phys)])
            elem_data.vel = vel
            elem_data.pres = pres
            elem_data.scal = vel[:1]

            elem_list.append(elem_data)

    hexa_data = MagicMock()
    hexa_data.ndim = ndim
    hexa_data.nel = nel
    hexa_data.lr1 = (nx1, ny1, nz1)
    hexa_data.var = ["u", "v", "w", "p", "t"]
    hexa_data.time = [0.0]
    hexa_data.istep = [1]
    hexa_data.elem = elem_list

    with patch("pymech.readnek", return_value=hexa_data):
        # Load data
        data = load_data(str(test_dir), loader_type="nek", field="u_1")

        q = data["q"]
        x = data["x"]
        y = data["y"]
        weights = data["spatial_weights"]
        Nspace = q.shape[1]

        assert x.shape == (Nspace,)
        assert y.shape == (Nspace,)
        assert weights.shape == (Nspace,)

        # Integral should still be pi^2/4
        f_values = q[0]
        integral = np.sum(weights * f_values**2)
        expected = (np.pi / 2.0) ** 2

        rel_error = np.abs(integral - expected) / expected
        print(f"Unequal elements: integral={integral:.15e}, expected={expected:.15e}, rel_error={rel_error:.2e}")

        assert rel_error < 1e-10, (
            f"Integral {integral} does not match expected {expected} (unequal elements); relative error {rel_error}"
        )


def test_nek_loader_without_pymech() -> None:
    """
    Test that a clear error is raised when pymech is missing.

    Use monkeypatch to make pymech import fail.
    """
    with patch.dict(sys.modules, {"pymech": None}):
        from openmodalpy.core.nek import NekDataLoader

        loader = NekDataLoader()
        with pytest.raises(ImportError, match="nek extra"):
            loader.load("/dummy/path", field="u_1")


def test_nek_loader_weights_failure(tmp_path: Path) -> None:
    """
    Test that incorrect weights produce a large integral error.

    Temporarily replace computed weights with all-ones, verify the test
    fails, then check that correct weights pass.
    """
    try:
        import pymech  # noqa: F401
    except ImportError:
        pytest.skip("pymech not installed")

    from openmodalpy.core.nek import gll_nodes_and_weights

    test_dir = tmp_path / "nek_test_weights"
    test_dir.mkdir()
    dummy_file = test_dir / "case.f00001"
    dummy_file.touch()

    # Setup: equal-element test data
    nx1, ny1, nel_x, nel_y = 5, 5, 2, 2
    nodes_gll, _ = gll_nodes_and_weights(nx1)
    nel = nel_x * nel_y
    nz1 = 1
    ndim = 2

    elem_list = []
    for iy in range(nel_y):
        y_start = 0.0 + iy * (np.pi / nel_y)
        y_end = y_start + (np.pi / nel_y)

        for ix in range(nel_x):
            x_start = 0.0 + ix * (np.pi / nel_x)
            x_end = x_start + (np.pi / nel_x)

            x_scale = (x_end - x_start) / 2.0
            y_scale = (y_end - y_start) / 2.0

            x_phys = np.zeros((nz1, ny1, nx1))
            y_phys = np.zeros((nz1, ny1, nx1))

            for j in range(ny1):
                for i in range(nx1):
                    x_phys[0, j, i] = x_start + x_scale * (nodes_gll[i] + 1.0)
                    y_phys[0, j, i] = y_start + y_scale * (nodes_gll[j] + 1.0)

            vel = np.zeros((3, nz1, ny1, nx1))
            vel[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            pres = np.zeros((1, nz1, ny1, nx1))
            pres[0, 0, :, :] = np.sin(x_phys[0]) * np.cos(y_phys[0])

            elem_data = MagicMock()
            elem_data.pos = np.array([x_phys, y_phys, np.zeros_like(x_phys)])
            elem_data.vel = vel
            elem_data.pres = pres
            elem_data.scal = vel[:1]

            elem_list.append(elem_data)

    hexa_data = MagicMock()
    hexa_data.ndim = ndim
    hexa_data.nel = nel
    hexa_data.lr1 = (nx1, ny1, nz1)
    hexa_data.var = ["u", "v", "w", "p", "t"]
    hexa_data.time = [0.0]
    hexa_data.istep = [1]
    hexa_data.elem = elem_list

    with patch("pymech.readnek", return_value=hexa_data):
        # First, load with correct weights and verify it passes
        data_correct = load_data(str(test_dir), loader_type="nek", field="u_1")
        q = data_correct["q"]
        weights = data_correct["spatial_weights"]
        f_values = q[0]
        integral_correct = np.sum(weights * f_values**2)
        expected = (np.pi / 2.0) ** 2
        rel_error_correct = np.abs(integral_correct - expected) / expected

        print(f"Correct weights: integral={integral_correct:.15e}, rel_error={rel_error_correct:.2e}")
        assert rel_error_correct < 1e-10

        # Now test with uniform weights (should fail)
        uniform_weights = np.ones_like(weights)
        integral_uniform = np.sum(uniform_weights * f_values**2)
        rel_error_uniform = np.abs(integral_uniform - expected) / expected

        print(f"Uniform weights: integral={integral_uniform:.15e}, rel_error={rel_error_uniform:.2e}")
        # Uniform weights should give a much larger error
        assert rel_error_uniform > 0.01, "Uniform weights should produce large error but did not"
