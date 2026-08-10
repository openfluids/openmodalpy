"""Regression tests for the FFT block cache stamp/verify mechanism.

These tests target `BaseAnalyzer._qhat_cache_stamp`/`_write_qhat_stamp`/
`_verify_qhat_stamp` (src/openmodalpy/core/base.py) as exercised through
`SPODAnalyzer.compute_fft_blocks` (src/openmodalpy/spod.py). Before the fix,
the cache key/validation ignored window_type and the content of `q`, so a
cache hit could silently serve blocks computed under different parameters or
from a different dataset. Each test below documents (in its docstring) the
failure mode it would have hit against the pre-fix code.
"""

import logging

import h5py
import numpy as np

from openmodalpy import SPODAnalyzer
from openmodalpy.core.base import blocksfft


def _make_data(q, dt=1.0):
    Ns, Nspace = q.shape
    Nx = Nspace
    Ny = 1
    return {
        "q": q,
        "x": np.linspace(0, 1, Nx),
        "y": np.linspace(0, 1, Ny),
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }


def _make_spod(tmp_path, q, *, nfft=8, overlap=0.0, window_type="hamming", file_path="dummy.h5"):
    data = _make_data(q)
    analyzer = SPODAnalyzer(
        file_path=file_path,
        nfft=nfft,
        overlap=overlap,
        window_type=window_type,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


def test_window_type_change_produces_different_qhat(tmp_path):
    """Same data/nfft/overlap/Ns, only window_type changes hamming -> hann.

    Pre-fix: the cache key/validation carried no window_type, so the second
    run silently returned the first run's hamming-windowed blocks and this
    assertion failed (qhat was bit-identical instead of different).
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal((32, 4))

    analyzer_hamming = _make_spod(tmp_path, q, window_type="hamming")
    analyzer_hann = _make_spod(tmp_path, q, window_type="hann")

    assert not analyzer_hamming.qhat_cached  # cold cache for the first run
    assert not np.allclose(analyzer_hamming.qhat, analyzer_hann.qhat)


def test_different_q_arrays_do_not_share_cache(tmp_path):
    """Two different `q` arrays sharing (data_root, nfft, overlap, Ns) must
    not serve each other's FFT blocks.

    Pre-fix: the cache path is built solely from (data_root, nfft, overlap,
    Ns, analysis_type) with no content check, so the second analyzer (over a
    completely different `q`) reused the first analyzer's cached blocks.
    """
    rng = np.random.default_rng(1)
    q1 = rng.standard_normal((32, 4))
    q2 = rng.standard_normal((32, 4)) * 5.0 + 3.0  # clearly distinct content

    analyzer1 = _make_spod(tmp_path, q1, file_path="dummy.h5")
    assert not analyzer1.qhat_cached

    analyzer2 = _make_spod(tmp_path, q2, file_path="dummy.h5")

    # Freshly computed reference for q2, independent of any cache.
    novlap = int(0.0 * 8)
    nblocks = (q2.shape[0] - novlap) // (8 - novlap)  # floor, matches welch_nblocks
    reference = blocksfft(
        q2,
        8,
        nblocks,
        novlap,
        blockwise_mean=False,
        normvar=False,
        window_norm="power",
        window_type="hamming",
    )

    np.testing.assert_allclose(analyzer2.qhat, reference)
    assert not np.allclose(analyzer2.qhat, analyzer1.qhat)


def test_stamp_mismatch_recomputes_without_raising(tmp_path, caplog):
    """A cache file whose stamped parameters disagree with the current
    analyzer must be rejected (recomputed), never raise, and never silently
    serve the mismatched blocks.

    Pre-fix: there was no stamp at all, so this scenario could not even be
    expressed — any existing file with the right shape was accepted.
    """
    rng = np.random.default_rng(2)
    q = rng.standard_normal((32, 4))

    _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    fname = "dummy_Nfft8_ovlap0.0_32snapshots_spod.hdf5"
    cache_file = tmp_path / fname
    assert cache_file.exists()

    # Corrupt the stamp in place to simulate a disagreeing/legacy cache file.
    with h5py.File(cache_file, "a") as f:
        f.attrs["_fftcache_window_type"] = "hann"

    # The mismatch notice moved from stdout to the module logger; assert on the
    # record so the check keeps its strength (message AND level), not just its text.
    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        analyzer2 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")

    assert not analyzer2.qhat_cached
    mismatch = [r for r in caplog.records if "FFT cache stamp mismatch" in r.getMessage()]
    assert mismatch, f"no stamp-mismatch warning logged; saw {[r.getMessage() for r in caplog.records]}"
    assert mismatch[0].levelno == logging.WARNING

    # The recomputed result must be correct (hamming), not the corrupted stamp's hann.
    novlap = 0
    nblocks = (q.shape[0] - novlap) // (8 - novlap)  # floor, matches welch_nblocks
    reference = blocksfft(
        q,
        8,
        nblocks,
        novlap,
        blockwise_mean=False,
        normvar=False,
        window_norm="power",
        window_type="hamming",
    )
    np.testing.assert_allclose(analyzer2.qhat, reference)


def test_matching_cache_hit_is_still_used(tmp_path, caplog):
    """A legitimate, fully-matching cache hit must still be served from disk.

    This is the mandatory counterpart to the tests above: a fix that merely
    disables caching (e.g. always recompute) would pass all of them while
    destroying the feature.
    """
    rng = np.random.default_rng(3)
    q = rng.standard_normal((32, 4))

    analyzer1 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    assert not analyzer1.qhat_cached

    with caplog.at_level(logging.INFO, logger="openmodalpy.core.base"):
        analyzer2 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")

    assert analyzer2.qhat_cached
    assert any("Loaded cached FFT blocks" in r.getMessage() for r in caplog.records)
    np.testing.assert_array_equal(analyzer2.qhat, analyzer1.qhat)


def test_corrupt_truncated_spod_cache_recomputes_without_raising(tmp_path, caplog):
    """A truncated (corrupt) SPOD cache must recompute and rewrite, not raise.

    Exercises the unreadable-file path (h5py OSError on open), not the
    stamp-mismatch path covered by ``test_stamp_mismatch_recomputes_without_raising``.
    """
    rng = np.random.default_rng(4)
    q = rng.standard_normal((32, 4))

    cold = _make_spod(tmp_path, q, file_path="dummy.h5")
    assert not cold.qhat_cached
    ref = np.array(cold.qhat, copy=True)

    fname = "dummy_Nfft8_ovlap0.0_32snapshots_spod.hdf5"
    cache_file = tmp_path / fname
    assert cache_file.exists()
    with open(cache_file, "r+b") as fh:
        fh.truncate(64)

    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        warm = _make_spod(tmp_path, q, file_path="dummy.h5")

    assert not warm.qhat_cached
    assert any("Failed to load cached FFT blocks" in r.getMessage() for r in caplog.records)
    np.testing.assert_allclose(warm.qhat, ref)

    # Third run must hit the rewritten cache.
    again = _make_spod(tmp_path, q, file_path="dummy.h5")
    assert again.qhat_cached
    np.testing.assert_allclose(again.qhat, ref)


def _make_bsmd(tmp_path, q, *, nfft=8, overlap=0.0, file_path="dummy.h5"):
    from openmodalpy import BSMDAnalyzer

    data = _make_data(q)
    analyzer = BSMDAnalyzer(
        file_path=file_path,
        nfft=nfft,
        overlap=overlap,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


def test_corrupt_truncated_bsmd_cache_recomputes_without_raising(tmp_path, caplog):
    """A truncated BSMD cache must recompute rather than raise on the read."""
    rng = np.random.default_rng(5)
    q = rng.standard_normal((32, 4))

    cold = _make_bsmd(tmp_path, q, file_path="dummy.h5")
    assert not cold.qhat_cached
    ref = np.array(cold.qhat, copy=True)

    fname = "dummy_Nfft8_ovlap0.0_32snapshots_bsmd.hdf5"
    cache_file = tmp_path / fname
    assert cache_file.exists()
    with open(cache_file, "r+b") as fh:
        fh.truncate(64)

    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        warm = _make_bsmd(tmp_path, q, file_path="dummy.h5")

    assert not warm.qhat_cached
    assert any("Failed to load cached FFT blocks" in r.getMessage() for r in caplog.records)
    np.testing.assert_allclose(warm.qhat, ref)

    again = _make_bsmd(tmp_path, q, file_path="dummy.h5")
    assert again.qhat_cached
    np.testing.assert_allclose(again.qhat, ref)


def test_bsmd_adopts_stamp_matching_spod_sibling(tmp_path):
    """BSMD serves from a stamp-matching SPOD cache in the same results_dir.

    Both write only their own ``..._<type>.hdf5``; adopting a sibling still
    copies the blocks into the reader's own cache file for later runs.
    """
    rng = np.random.default_rng(6)
    q = rng.standard_normal((32, 4))

    spod = _make_spod(tmp_path, q, file_path="dummy.h5")
    assert not spod.qhat_cached
    spod_name = "dummy_Nfft8_ovlap0.0_32snapshots_spod.hdf5"
    bsmd_name = "dummy_Nfft8_ovlap0.0_32snapshots_bsmd.hdf5"
    assert (tmp_path / spod_name).exists()
    assert not (tmp_path / bsmd_name).exists()

    bsmd = _make_bsmd(tmp_path, q, file_path="dummy.h5")
    assert bsmd.qhat_cached
    np.testing.assert_array_equal(bsmd.qhat, spod.qhat)
    # Adoption writes a local BSMD copy; never renames or overwrites the SPOD file.
    assert (tmp_path / bsmd_name).exists()
    assert (tmp_path / spod_name).exists()

    warm = _make_bsmd(tmp_path, q, file_path="dummy.h5")
    assert warm.qhat_cached
    np.testing.assert_array_equal(warm.qhat, spod.qhat)


def test_corrupt_sibling_cache_recomputes_without_raising(tmp_path, caplog):
    """A truncated sibling cache must recompute, with the file named in the warning.

    Lookup tries the reader's own file first, then siblings in the same
    ``results_dir``. An unreadable sibling must fail soft — same policy as a
    corrupt own-file read.
    """
    rng = np.random.default_rng(7)
    q = rng.standard_normal((32, 4))

    # Cold BSMD with no sibling present (reference blocks).
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    cold = _make_bsmd(ref_dir, q, file_path="dummy.h5")
    assert not cold.qhat_cached
    ref = np.array(cold.qhat, copy=True)

    shared = tmp_path / "shared"
    shared.mkdir()
    _make_spod(shared, q, file_path="dummy.h5")
    spod_name = "dummy_Nfft8_ovlap0.0_32snapshots_spod.hdf5"
    spod_cache = shared / spod_name
    assert spod_cache.exists()
    with open(spod_cache, "r+b") as fh:
        fh.truncate(64)

    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        warm = _make_bsmd(shared, q, file_path="dummy.h5")

    assert not warm.qhat_cached
    failure = [r for r in caplog.records if "Failed to load cached FFT blocks" in r.getMessage()]
    assert failure, f"no load-failure warning; saw {[r.getMessage() for r in caplog.records]}"
    assert str(spod_cache) in failure[0].getMessage()
    np.testing.assert_allclose(warm.qhat, ref)


def _sibling_path(tmp_path, analysis):
    return tmp_path / f"dummy_Nfft8_ovlap0.0_32snapshots_{analysis}.hdf5"


def test_saving_fft_blocks_without_q_warns_that_the_next_run_recomputes(tmp_path, caplog):
    """Saving FFTBlocks without source snapshots must warn that the next run recomputes.

    The cache stamp is derived from ``q``. When blocks are written but ``q`` is
    gone, the file holds unstamped FFTBlocks that no later run can validate.
    """
    rng = np.random.default_rng(12)
    q = rng.standard_normal((32, 4))
    analyzer = _make_spod(tmp_path, q, file_path="dummy.h5")
    # Simulate a save after the source snapshots have left memory.
    analyzer.data = {k: v for k, v in analyzer.data.items() if k != "q"}
    assert analyzer.data.get("q") is None
    assert analyzer.qhat is not None and analyzer.qhat.size > 0

    with caplog.at_level(logging.WARNING, logger="openmodalpy.spod"):
        analyzer.save_results()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "recomput" in r.getMessage().lower()]
    assert warnings, f"expected a WARNING about recomputing; saw {[r.getMessage() for r in caplog.records]}"


def test_saving_fft_blocks_with_q_does_not_warn(tmp_path, caplog):
    """Saving FFTBlocks while ``q`` is still in memory must not emit the no-stamp warning.

    Counterpart to ``test_saving_fft_blocks_without_q_warns_that_the_next_run_recomputes``:
    without this, an unconditional warning would make that test pass spuriously.
    """
    rng = np.random.default_rng(13)
    q = rng.standard_normal((32, 4))
    analyzer = _make_spod(tmp_path, q, file_path="dummy.h5")
    assert analyzer.data.get("q") is not None
    assert analyzer.qhat is not None and analyzer.qhat.size > 0

    with caplog.at_level(logging.WARNING, logger="openmodalpy.spod"):
        analyzer.save_results()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "recomput" in r.getMessage().lower()]
    assert not warnings, f"unexpected recompute warning(s): {[r.getMessage() for r in warnings]}"


def test_malformed_sibling_cache_recomputes_instead_of_raising(tmp_path, caplog):
    """A neighbour's bad cache file must never abort this analyzer's run.

    Sibling adoption opens files written by OTHER analyses, so a malformed one
    is no longer only the reader's own problem. Two payloads that are readable
    as HDF5 but wrong: FFTBlocks stored with the wrong rank, and a stamp
    attribute that will not cast to its expected type. Both used to escape the
    OSError-only guard as IndexError and ValueError respectively.
    """
    from openmodalpy.core.base import _write_qhat_stamp

    rng = np.random.default_rng(11)
    q = rng.standard_normal((32, 8))
    reference = _make_spod(tmp_path / "ref", q, file_path="dummy.h5").qhat

    for case, mutate in (
        ("wrong_rank", "rank"),
        ("bad_stamp_attr", "stamp"),
    ):
        case_dir = tmp_path / case
        case_dir.mkdir()
        stamped = _make_spod(case_dir / "stamp_src", q, file_path="dummy.h5")

        sibling = _sibling_path(case_dir, "bsmd")
        with h5py.File(sibling, "w") as f:
            blocks = np.ones((5, 8), dtype=complex) if mutate == "rank" else np.ones((5, 8, 3), dtype=complex)
            f.create_dataset("FFTBlocks", data=blocks)
            _write_qhat_stamp(f, stamped, _make_data(q)["q"])
            if mutate == "stamp":
                for key in list(f.attrs):
                    if key.endswith("nfft"):
                        f.attrs[key] = "not-an-int"

        with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
            analyzer = _make_spod(case_dir, q, file_path="dummy.h5")

        assert not analyzer.qhat_cached, f"{case}: adopted a malformed sibling"
        np.testing.assert_allclose(analyzer.qhat, reference, err_msg=case)
