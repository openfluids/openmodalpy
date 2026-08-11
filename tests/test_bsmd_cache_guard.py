"""BSMD FFT-cache OSError guards cover reads only.

A failed cache WRITE must propagate and must not be printed as a load failure.
A failed cache READ must still recompute and name the file.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer

NFFT = 8
NS = 32
NSPACE = 4


def _data() -> dict:
    q = np.random.default_rng(0).standard_normal((NS, NSPACE))
    return {
        "q": q,
        "x": np.linspace(0.0, 1.0, NSPACE),
        "y": np.linspace(0.0, 1.0, 1),
        "dt": 1.0,
        "Nx": NSPACE,
        "Ny": 1,
        "Ns": NS,
    }


def _bsmd(results_dir) -> BSMDAnalyzer:
    return BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=NFFT,
        overlap=0.0,
        results_dir=str(results_dir),
        figures_dir=str(results_dir),
        data_loader=lambda _: _data(),
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=[(0, 0, 0)],
        use_parallel=False,
    )


def test_write_failure_is_not_called_a_load_failure(tmp_path, monkeypatch, caplog):
    """A full disk while SAVING the BSMD cache must not log a load failure."""
    import logging

    import openmodalpy.core.base as base_mod

    bsmd_dir = tmp_path / "bsmd"
    bsmd_dir.mkdir(parents=True, exist_ok=True)

    real_file = h5py.File

    def fake_file(path, mode="r", *args, **kwargs):
        if str(path).startswith(str(bsmd_dir)) and mode != "r":
            raise OSError("No space left on device")
        return real_file(path, mode, *args, **kwargs)

    # Cache load/save lives in BaseAnalyzer; patch the module that opens the file.
    monkeypatch.setattr(base_mod.h5py, "File", fake_file)

    analyzer = _bsmd(bsmd_dir)
    analyzer.load_and_preprocess()
    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        with pytest.raises(OSError, match="No space left on device"):
            analyzer.compute_fft_blocks()
    load_msgs = [r.getMessage() for r in caplog.records if "Failed to load cached FFT blocks" in r.getMessage()]
    assert not load_msgs, "a cache WRITE failure was reported as a cache LOAD failure:\n" + "\n".join(load_msgs)


def test_read_failure_still_recomputes_and_names_the_file(tmp_path, caplog):
    """A corrupt BSMD cache must still recompute, with the file named."""
    import logging

    bsmd_dir = tmp_path / "bsmd"
    bsmd_dir.mkdir(parents=True, exist_ok=True)
    analyzer = _bsmd(bsmd_dir)
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    cache_path = analyzer._qhat_cache_path
    assert cache_path is not None

    with open(cache_path, "wb") as fh:
        fh.write(b"not an hdf5 file at all")

    fresh = _bsmd(bsmd_dir)
    fresh.load_and_preprocess()
    with caplog.at_level(logging.WARNING, logger="openmodalpy.core.base"):
        fresh.compute_fft_blocks()
    # The failure record itself must name the file.
    failure = [r for r in caplog.records if "Failed to load cached FFT blocks" in r.getMessage()]
    assert len(failure) == 1, [r.getMessage() for r in caplog.records]
    assert str(cache_path) in failure[0].getMessage()
    assert fresh.qhat_cached is False
    assert fresh.qhat.shape[0] == NFFT // 2 + 1


def test_cache_path_is_none_before_any_fft(tmp_path):
    """A new analyzer must already carry ``_qhat_cache_path``, set to None.

    ``compute_fft_blocks`` only assigns this attribute when a cache file is in
    use, so the readers in ``_maybe_offload_qhat`` and ``save_results`` test it
    for None. If nothing declares it up front, those reads raise AttributeError
    instead on any run without a cache.
    """
    analyzer = _bsmd(tmp_path)
    assert analyzer._qhat_cache_path is None
