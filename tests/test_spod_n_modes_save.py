"""SPOD keeps only the leading ``n_modes_save`` modes at each frequency.

Without ``n_modes_save`` SPOD keeps one mode per Welch block at every
frequency, so ``modes`` is ``(n_freq, n_space, n_blocks)``. On a large record
that array is the largest thing SPOD makes. These checks pin what
``n_modes_save`` truncates, what it leaves alone, and that the modes it keeps
are the leading ones.

The block axis and the mode axis of ``time_coefficients`` have the same length
when nothing is truncated, so a check that only reads shapes cannot tell them
apart. The checks below truncate to a count below the block count, which makes
the two axes different lengths and gives the axis order a way to go red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openmodalpy import SPODAnalyzer

N_SPACE = 12
NFFT = 16
NBLOCKS = 6
DT = 0.05
N_KEEP = 2


def _field() -> dict[str, object]:
    """Seeded record with a full-rank block Gram at every frequency.

    A full-rank Gram gives ``NBLOCKS`` distinct eigenvalues per frequency, so
    the leading two modes are unambiguous and truncation has something to cut.
    """
    rng = np.random.default_rng(20260831)
    ns = NBLOCKS * NFFT
    return {
        "q": rng.standard_normal((ns, N_SPACE)),
        "x": np.arange(N_SPACE, dtype=float),
        "y": np.array([0.0]),
        "dt": DT,
        "Nx": N_SPACE,
        "Ny": 1,
        "Ns": ns,
    }


def _run(tmp_path: Path, *, n_modes_save: int | None = None) -> SPODAnalyzer:
    """Drive SPOD end to end on the seeded record; output lands in ``tmp_path``."""
    field = _field()
    kwargs: dict[str, object] = {}
    if n_modes_save is not None:
        kwargs["n_modes_save"] = n_modes_save
    analyzer = SPODAnalyzer(
        file_path="spod_n_modes_save",
        nfft=NFFT,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: field,
        spatial_weights=np.ones((N_SPACE, 1)),
        **kwargs,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_spod()
    return analyzer


def test_default_keeps_every_mode(tmp_path: Path) -> None:
    """No ``n_modes_save`` keeps one mode per block, as before this option existed."""
    analyzer = _run(tmp_path)

    assert analyzer.nblocks == NBLOCKS
    assert np.asarray(analyzer.modes).shape[-1] == NBLOCKS
    assert np.asarray(analyzer.eigenvalues).shape[-1] == NBLOCKS
    assert np.asarray(analyzer.time_coefficients).shape[-1] == NBLOCKS


def test_n_modes_save_truncates_modes_but_not_eigenvalues(tmp_path: Path) -> None:
    """Modes and time coefficients cut to ``k``; the spectrum keeps every block."""
    analyzer = _run(tmp_path, n_modes_save=N_KEEP)

    modes = np.asarray(analyzer.modes)
    coefficients = np.asarray(analyzer.time_coefficients)
    eigenvalues = np.asarray(analyzer.eigenvalues)
    n_freq = modes.shape[0]

    assert modes.shape == (n_freq, N_SPACE, N_KEEP)
    # Axis 1 is the block axis and stays full; axis 2 is the mode axis and cuts.
    assert coefficients.shape == (n_freq, NBLOCKS, N_KEEP)
    # The spectrum figure plots energy per block, so eigenvalues keep every one.
    assert eigenvalues.shape == (n_freq, NBLOCKS)


@pytest.mark.characterization
def test_kept_modes_are_the_leading_ones(tmp_path: Path) -> None:
    """Truncation keeps the first ``k`` modes of the untruncated run, unchanged."""
    full = _run(tmp_path / "full")
    cut = _run(tmp_path / "cut", n_modes_save=N_KEEP)

    np.testing.assert_allclose(
        np.asarray(cut.modes),
        np.asarray(full.modes)[:, :, :N_KEEP],
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(cut.time_coefficients),
        np.asarray(full.time_coefficients)[:, :, :N_KEEP],
        rtol=1e-12,
        atol=0.0,
    )


def test_more_modes_than_blocks_warns_and_clamps(tmp_path: Path) -> None:
    """Asking for more modes than blocks reports both counts and keeps all blocks."""
    too_many = NBLOCKS + 3
    with pytest.warns(RuntimeWarning, match=rf"{too_many}.*{NBLOCKS}"):
        analyzer = _run(tmp_path, n_modes_save=too_many)

    assert np.asarray(analyzer.modes).shape[-1] == NBLOCKS


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_n_modes_save_raises(tmp_path: Path, bad: int) -> None:
    """A mode count below one is refused where it is given, not later."""
    with pytest.raises(ValueError, match="n_modes_save"):
        SPODAnalyzer(
            file_path="spod_n_modes_save",
            nfft=NFFT,
            overlap=0.0,
            results_dir=str(tmp_path),
            figures_dir=str(tmp_path),
            data_loader=lambda _: _field(),
            n_modes_save=bad,
        )


def test_truncated_result_saves_and_reloads(tmp_path: Path) -> None:
    """A truncated run round-trips through the results file with its own shapes."""
    analyzer = _run(tmp_path, n_modes_save=N_KEEP)
    saved_modes = np.asarray(analyzer.modes).copy()
    analyzer.save_results()

    reloaded = _run(tmp_path, n_modes_save=N_KEEP)
    reloaded.modes = np.array([])
    reloaded.load_results()

    np.testing.assert_allclose(np.asarray(reloaded.modes), saved_modes, rtol=1e-12, atol=0.0)
