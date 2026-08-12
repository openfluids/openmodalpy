"""n_modes_saved must describe the file; load must not discard modes (openmodalpy-e1y)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from openmodalpy import MPODAnalyzer, PODAnalyzer, STPODAnalyzer


def _full_rank_field(n_snap: int = 24, n_space: int = 16, seed: int = 1) -> dict:
    """Full-rank-ish field so n_modes_save actually controls stored width."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_snap, n_space))
    nx = int(np.sqrt(n_space))
    ny = n_space // nx
    assert nx * ny == n_space
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.linspace(0.0, 1.0, ny),
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": n_snap,
    }


def _make(cls, tmp_path, *, n_modes_save: int, name: str, kwargs: dict | None = None):
    data = _full_rank_field()
    return cls(
        file_path=name,
        n_modes_save=n_modes_save,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        **(kwargs or {}),
    )


@pytest.mark.parametrize(
    "cls, kwargs, run",
    [
        (PODAnalyzer, {}, "perform_pod"),
        (STPODAnalyzer, {"embedding_dim": 3}, "perform_stpod"),
        # mPOD gets both fixes only through POD's save/load; prove the
        # inheritance actually carries them.
        (
            MPODAnalyzer,
            {"band_edges": [0.0, 0.25, 0.5, 1.0], "band_scale": "normalized_nyquist"},
            "perform_mpod",
        ),
    ],
    ids=["pod", "stpod", "mpod-multiband"],
)
def test_load_narrow_file_into_wide_cap_honest_resave(cls, kwargs, run, tmp_path):
    """3-mode file into n_modes_save=12: counter falls; resave declares 3."""
    write_dir = tmp_path / "w"
    read_dir = tmp_path / "r"
    writer = _make(cls, write_dir, n_modes_save=3, name="narrow", kwargs=kwargs)
    writer.load_and_preprocess()
    getattr(writer, run)()
    assert writer.modes.shape[1] == 3
    writer.save_results("narrow.hdf5")

    reader = _make(cls, read_dir, n_modes_save=12, name="wide_cap", kwargs=kwargs)
    reader.load_results(str(write_dir / "narrow.hdf5"))

    loaded_width = int(reader.modes.shape[1])
    assert loaded_width == 3, f"loaded modes discarded or padded: got width {loaded_width}"
    assert reader.n_modes_save == 3, f"wide cap must fall to loaded width; got n_modes_save={reader.n_modes_save}"

    reader.save_results("resave.hdf5")
    with h5py.File(read_dir / "resave.hdf5") as handle:
        declared = int(handle.attrs["n_modes_saved"])
        stored = int(handle["modes"].shape[1])
    assert stored == 3, f"resave must keep all loaded modes; held {stored}"
    assert declared == stored, f"file declares {declared} modes but holds {stored}"


@pytest.mark.parametrize(
    "cls, kwargs, run",
    [
        (PODAnalyzer, {}, "perform_pod"),
        (STPODAnalyzer, {"embedding_dim": 3}, "perform_stpod"),
        # mPOD gets both fixes only through POD's save/load; prove the
        # inheritance actually carries them.
        (
            MPODAnalyzer,
            {"band_edges": [0.0, 0.25, 0.5, 1.0], "band_scale": "normalized_nyquist"},
            "perform_mpod",
        ),
    ],
    ids=["pod", "stpod", "mpod-multiband"],
)
def test_load_wide_file_into_narrow_cap_keeps_modes(cls, kwargs, run, tmp_path):
    """12-mode file into n_modes_save=5: modes stay 12; cap stays 5; resave declares 12."""
    write_dir = tmp_path / "w"
    read_dir = tmp_path / "r"
    writer = _make(cls, write_dir, n_modes_save=12, name="wide", kwargs=kwargs)
    writer.load_and_preprocess()
    getattr(writer, run)()
    written_width = int(writer.modes.shape[1])
    assert written_width >= 12, f"writer must store a wide set; got {written_width}"
    writer.save_results("wide.hdf5")

    reader = _make(cls, read_dir, n_modes_save=5, name="narrow_cap", kwargs=kwargs)
    reader.load_results(str(write_dir / "wide.hdf5"))

    loaded_width = int(reader.modes.shape[1])
    assert loaded_width == written_width, (
        f"narrow cap must not drop loaded modes: held {loaded_width}, file had {written_width}"
    )
    assert reader.n_modes_save == 5, f"narrow cap must not rise on load; got n_modes_save={reader.n_modes_save}"

    reader.save_results("resave.hdf5")
    with h5py.File(read_dir / "resave.hdf5") as handle:
        declared = int(handle.attrs["n_modes_saved"])
        stored = int(handle["modes"].shape[1])
    assert stored == written_width, f"resave must keep all loaded modes; held {stored}"
    assert declared == stored, f"file declares {declared} modes but holds {stored}"
