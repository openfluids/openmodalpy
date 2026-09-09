#!/usr/bin/env python3
"""
Tests for the Nek5000 field reader and its quadrature weights.

Each test writes a real Nek5000 field file with pymech and reads it back
through ``load_data``. The field is sin(x) cos(y) on the box [0, pi]^2, so
the weighted energy sum(w q^2) has the closed-form value pi^2 / 4. A wrong
quadrature weight, a wrong element Jacobian, or a wrong flattening order all
move that number.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from openmodalpy.core.io import load_data
from openmodalpy.core.nek import gll_nodes_and_weights

# The integral of sin(x)^2 cos(y)^2 over [0, pi] x [0, pi].
EXPECTED_ENERGY = (np.pi / 2.0) ** 2

# Nine points a side. GLL quadrature is exact for polynomials only, and
# sin(x)^2 is not one, so the order sets the accuracy floor: five points give
# 2.0e-05 on this mesh and nine give 5.2e-14. At nine points the error that
# remains comes from the mesh mapping, not from the polynomial order.
ORDER = 9


def _write_nek_case(
    directory: Path,
    *,
    element_widths_x: list[float],
    nel_y: int = 2,
    order: int = ORDER,
    times: tuple[float, ...] = (0.0,),
) -> None:
    """
    Write one Nek5000 field file per time to a directory.

    The mesh covers [0, pi] x [0, pi]. The elements divide x by the widths
    given, which do not have to be equal, and divide y evenly. The x velocity
    holds sin(x) cos(y) scaled by the index of the snapshot plus one, the y
    velocity holds cos(x) sin(y), and the pressure holds x + 2y.

    Args:
        directory (Path): Where to write the files.
        element_widths_x (list[float]): Element widths in x. They must add to pi.
        nel_y (int): Number of elements in y.
        order (int): Points per element in each direction.
        times (tuple[float, ...]): Time stamp of each file.
    """
    pymech = pytest.importorskip("pymech")
    from pymech.core import HexaData

    assert sum(element_widths_x) == pytest.approx(np.pi)

    nodes, _ = gll_nodes_and_weights(order)
    nel_x = len(element_widths_x)
    nel = nel_x * nel_y
    nz1 = 1

    for snapshot_index, time in enumerate(times):
        # var counts: (positions, velocities, pressures, temperatures, scalars).
        data = HexaData(2, nel, (order, order, nz1), (2, 2, 1, 1, 0))
        data.wdsz = 8
        data.endian = "little"
        data.time = time
        data.istep = snapshot_index + 1

        element = 0
        for iy in range(nel_y):
            y_start = iy * (np.pi / nel_y)
            y_half_width = (np.pi / nel_y) / 2.0
            for ix in range(nel_x):
                x_start = sum(element_widths_x[:ix])
                x_half_width = element_widths_x[ix] / 2.0

                x_phys = x_start + x_half_width * (nodes[None, None, :] + 1.0) * np.ones((nz1, order, 1))
                y_phys = y_start + y_half_width * (nodes[None, :, None] + 1.0) * np.ones((nz1, 1, order))

                elem = data.elem[element]
                element += 1
                elem.pos[0] = x_phys
                elem.pos[1] = y_phys
                elem.pos[2] = 0.0
                # Each stored field is different, so a test that asks for one
                # and gets another fails instead of passing on a look-alike.
                elem.vel[0] = (snapshot_index + 1) * np.sin(x_phys) * np.cos(y_phys)
                elem.vel[1] = np.cos(x_phys) * np.sin(y_phys)
                elem.pres[0] = x_phys + 2.0 * y_phys
                elem.temp[0] = 0.0

        pymech.writenek(str(directory / f"case0.f{snapshot_index + 1:05d}"), data)


def test_nek_round_trip_on_equal_elements(tmp_path: Path) -> None:
    """
    Test the reader on a mesh of four equal elements.

    The weighted energy must match pi^2 / 4 to spectral accuracy, and the
    coordinates must stay inside the box.
    """
    _write_nek_case(tmp_path, element_widths_x=[np.pi / 2.0, np.pi / 2.0])

    data = load_data(str(tmp_path), loader_type="nek", field="u_1")

    q = data["q"]
    x = data["x"]
    y = data["y"]
    weights = data["spatial_weights"]

    assert q.ndim == 2
    assert q.shape[0] == 1
    n_space = q.shape[1]
    assert n_space == 4 * ORDER * ORDER
    assert x.shape == (n_space,)
    assert y.shape == (n_space,)
    assert weights.shape == (n_space,)
    assert data["spatial_weight_type"] == "prescribed"

    # The weights must integrate the constant 1 over the box, which is pi^2.
    assert np.sum(weights) == pytest.approx(np.pi**2, rel=1e-12)

    energy = float(np.sum(weights * q[0] ** 2))
    assert energy == pytest.approx(EXPECTED_ENERGY, rel=1e-10)

    assert np.all(x >= -1e-12) and np.all(x <= np.pi + 1e-12)
    assert np.all(y >= -1e-12) and np.all(y <= np.pi + 1e-12)

    # The field must round-trip point by point, which pins the flattening order.
    np.testing.assert_allclose(q[0], np.sin(x) * np.cos(y), atol=1e-12)


def test_nek_round_trip_on_unequal_elements(tmp_path: Path) -> None:
    """
    Test the reader on a mesh whose elements have different widths.

    The energy is the same number, so this catches a Jacobian that is computed
    once and reused, or scaled by the wrong element width.
    """
    _write_nek_case(tmp_path, element_widths_x=[2.0 * np.pi / 3.0, np.pi / 3.0])

    data = load_data(str(tmp_path), loader_type="nek", field="u_1")

    q = data["q"]
    x = data["x"]
    y = data["y"]
    weights = data["spatial_weights"]
    n_space = q.shape[1]

    assert x.shape == (n_space,)
    assert y.shape == (n_space,)
    assert weights.shape == (n_space,)

    assert np.sum(weights) == pytest.approx(np.pi**2, rel=1e-12)

    energy = float(np.sum(weights * q[0] ** 2))
    assert energy == pytest.approx(EXPECTED_ENERGY, rel=1e-10)

    np.testing.assert_allclose(q[0], np.sin(x) * np.cos(y), atol=1e-12)


def test_nek_reads_a_sequence_of_snapshots(tmp_path: Path) -> None:
    """
    Test that several files become several snapshots in file order.

    The reader must take the time of each file and get the step from those
    times.
    """
    times = (0.0, 0.25, 0.5, 0.75, 1.0)
    _write_nek_case(tmp_path, element_widths_x=[np.pi / 2.0, np.pi / 2.0], order=5, times=times)

    data = load_data(str(tmp_path), loader_type="nek", field="u_1")

    q = data["q"]
    assert q.shape[0] == len(times)
    np.testing.assert_allclose(data["t"], np.array(times))
    assert data["dt"] == pytest.approx(0.25)

    # Snapshot k holds (k + 1) times the field of snapshot 0.
    for k in range(1, len(times)):
        np.testing.assert_allclose(q[k], (k + 1) * q[0], atol=1e-12)


def test_uniform_weights_would_get_the_energy_wrong(tmp_path: Path) -> None:
    """
    Test that the computed weights are not interchangeable with ones.

    This states what the weights buy. Replacing them with ones must move the
    energy far away from the closed-form value.
    """
    _write_nek_case(tmp_path, element_widths_x=[np.pi / 2.0, np.pi / 2.0])

    data = load_data(str(tmp_path), loader_type="nek", field="u_1")
    q = data["q"]
    weights = data["spatial_weights"]

    correct = float(np.sum(weights * q[0] ** 2))
    uniform = float(np.sum(np.ones_like(weights) * q[0] ** 2))

    assert correct == pytest.approx(EXPECTED_ENERGY, rel=1e-10)
    assert abs(uniform - EXPECTED_ENERGY) / EXPECTED_ENERGY > 1.0


def test_nek_loader_without_pymech() -> None:
    """Test that a missing pymech gives an error that names the extra."""
    with patch.dict(sys.modules, {"pymech": None}):
        from openmodalpy.core.nek import NekDataLoader

        loader = NekDataLoader()
        with pytest.raises(ImportError, match="nek extra"):
            loader.load("/dummy/path", field="u_1")


@pytest.mark.parametrize("n", [3, 5, 7, 9, 15])
def test_gll_nodes_and_weights_are_real(n: int) -> None:
    """
    Test that the GLL nodes and weights are real numbers.

    The roots of the Legendre derivative are real, but the solver can return
    them in a complex array. A complex node makes every later array complex,
    and the assignment into a float mesh then discards the imaginary part
    without a word. NumPy 2.5 showed this where NumPy 2.4 did not.
    """
    nodes, weights = gll_nodes_and_weights(n)

    assert not np.iscomplexobj(nodes), f"GLL nodes for n={n} are complex"
    assert not np.iscomplexobj(weights), f"GLL weights for n={n} are complex"

    # The nodes hold the two endpoints and rise from -1 to 1.
    assert nodes[0] == pytest.approx(-1.0)
    assert nodes[-1] == pytest.approx(1.0)
    assert np.all(np.diff(nodes) > 0.0)

    # The weights integrate the constant 1 over [-1, 1], which is 2.
    assert np.sum(weights) == pytest.approx(2.0, rel=1e-13)


def test_gll_nodes_drop_rounding_level_imaginary_parts() -> None:
    """
    Test that a complex root array with rounding-level noise is accepted.

    This is what NumPy 2.5 returns. The nodes must come back real and correct.
    """
    from openmodalpy.core import nek

    n = 5
    expected_nodes, expected_weights = nek.gll_nodes_and_weights(n)

    real_roots = nek.legendre.legroots(nek.legendre.legder(np.eye(n)[n - 1]))
    noisy = real_roots.astype(np.complex128) + 1e-17j

    with patch.object(nek.legendre, "legroots", return_value=noisy):
        nodes, weights = nek.gll_nodes_and_weights(n)

    assert not np.iscomplexobj(nodes)
    np.testing.assert_allclose(nodes, expected_nodes, atol=1e-14)
    np.testing.assert_allclose(weights, expected_weights, atol=1e-14)


def test_gll_nodes_reject_a_truly_complex_root() -> None:
    """
    Test that a root with a real imaginary part raises instead of casting.

    A silent cast would hide a broken solve behind plausible numbers.
    """
    from openmodalpy.core import nek

    n = 5
    real_roots = nek.legendre.legroots(nek.legendre.legder(np.eye(n)[n - 1]))
    broken = real_roots.astype(np.complex128)
    broken[0] += 0.5j

    with patch.object(nek.legendre, "legroots", return_value=broken):
        with pytest.raises(ValueError, match="came out complex"):
            nek.gll_nodes_and_weights(n)


def test_each_field_file_is_read_once(tmp_path: Path) -> None:
    """
    Test that the reader opens each file one time.

    The mesh and the metadata both come from the first file. Taking them with
    a second and a third read of that file costs the same again on a large
    case, and the cost is invisible in the result.
    """
    pymech = pytest.importorskip("pymech")

    times = (0.0, 0.25, 0.5)
    _write_nek_case(tmp_path, element_widths_x=[np.pi / 2.0, np.pi / 2.0], order=5, times=times)

    real_readnek = pymech.readnek
    calls: list[str] = []

    def counting_readnek(path: str, *args: object, **kwargs: object) -> object:
        calls.append(str(path))
        return real_readnek(path, *args, **kwargs)

    with patch.object(pymech, "readnek", counting_readnek):
        load_data(str(tmp_path), loader_type="nek", field="u_1")

    assert len(calls) == len(times), f"Expected one read per file, got {len(calls)}: {calls}"
    assert len(set(calls)) == len(times)


def test_files_without_a_time_leave_the_step_undefined(tmp_path: Path) -> None:
    """
    Test a run whose files all carry the time 0.0.

    Nek5000 stamps a time in each field file, but a file can be written
    without one. Every snapshot then shares the time 0.0, and the step between
    them is undefined. The reader must report dt as None and pass the times
    through unchanged. A made-up step of 1 would put every frequency this
    package reports on the wrong axis, and nothing downstream would say so.
    """
    _write_nek_case(
        tmp_path,
        element_widths_x=[np.pi / 2.0, np.pi / 2.0],
        order=5,
        times=(0.0, 0.0, 0.0, 0.0),
    )

    data = load_data(str(tmp_path), loader_type="nek", field="u_1")

    np.testing.assert_allclose(data["t"], np.zeros(4))
    assert data["dt"] is None


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("u_1", lambda x, y: np.sin(x) * np.cos(y)),
        ("u_2", lambda x, y: np.cos(x) * np.sin(y)),
        ("p", lambda x, y: x + 2.0 * y),
    ],
)
def test_the_field_argument_picks_the_right_array(
    tmp_path: Path,
    field: str,
    expected: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """
    Test that each field name reads the array it names.

    The three stored fields differ, so a reader that takes the pressure when
    asked for the second velocity component fails here.
    """
    _write_nek_case(tmp_path, element_widths_x=[np.pi / 2.0, np.pi / 2.0], order=5)

    data = load_data(str(tmp_path), loader_type="nek", field=field)

    x = data["x"]
    y = data["y"]
    np.testing.assert_allclose(data["q"][0], expected(x, y), atol=1e-12)


def test_an_unknown_field_name_is_refused(tmp_path: Path) -> None:
    """Test that a name the reader does not support gives a clear error."""
    from openmodalpy.core.nek import NekDataLoader

    with pytest.raises(ValueError, match="Unsupported variable"):
        NekDataLoader()._parse_field("t")
