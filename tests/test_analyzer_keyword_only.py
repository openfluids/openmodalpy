"""Pin the keyword-only constructor contract for every analyzer.

Every analyzer takes ``file_path`` positionally and every other parameter
by keyword. A positional call past ``file_path`` must raise ``TypeError``
immediately, instead of silently binding to the wrong parameter.
"""

import numpy as np
import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    PSDPODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
)
from openmodalpy.core.base import BaseAnalyzer

# Python says "takes from 1 to 2 positional arguments but N were given" when a
# keyword-only parameter is given by position. Pin that text. A bare TypeError
# is not enough: before this contract, PSDPODAnalyzer("data.mat", 256, 0.5)
# already raised TypeError, from os.makedirs receiving the int 256 as a
# directory name. A test that accepts any TypeError passes on both, so it
# cannot fail.
_POSITIONAL_TYPEERROR = r"positional argument"


def _welch_field(Ns: int = 32, Nspace: int = 6) -> dict:
    """Small analytic field with enough snapshots to form a few FFT blocks."""
    t = np.linspace(0.0, 2.0 * np.pi, Ns, endpoint=False)
    x = np.linspace(0.0, 1.0, Nspace)
    q = np.outer(np.sin(t), np.sin(2.0 * np.pi * x))
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": x,
        "y": np.array([0.0]),
        "dt": 0.1,
        "Nx": Nspace,
        "Ny": 1,
        "Ns": Ns,
    }


# (analyzer class, keyword-only kwargs that exercise its own parameters)
_CASES = [
    (BaseAnalyzer, {"spatial_weight_type": "uniform"}),
    (PODAnalyzer, {"n_modes_save": 3}),
    (MPODAnalyzer, {"n_modes_save": 3}),
    (SPODAnalyzer, {"nfft": 8, "overlap": 0.5}),
    (STPODAnalyzer, {"embedding_dim": 2, "n_modes_save": 2}),
    (DMDAnalyzer, {"rank": 2}),
    (BSMDAnalyzer, {"nfft": 8, "overlap": 0.5}),
    (PSDPODAnalyzer, {"nfft": 8, "overlap": 0.5}),
]


@pytest.mark.parametrize("cls,kwargs", _CASES)
def test_positional_call_past_file_path_raises_typeerror(cls, kwargs):
    """A second positional argument after file_path is rejected outright."""
    data = _welch_field()
    with pytest.raises(TypeError, match=_POSITIONAL_TYPEERROR):
        cls("dummy.h5", 8, data=data, **kwargs)


@pytest.mark.parametrize("cls,kwargs", _CASES)
def test_keyword_form_still_works(cls, kwargs):
    """The same options given as keywords construct the analyzer fine."""
    data = _welch_field()
    analyzer = cls(data=data, **kwargs)
    assert analyzer.data is data


def test_psd_pod_old_dangerous_positional_form_now_raises():
    """The historically silent PSDPODAnalyzer(path, 256, 0.5) call now raises."""
    with pytest.raises(TypeError, match=_POSITIONAL_TYPEERROR):
        PSDPODAnalyzer("data.mat", 256, 0.5)


def test_spod_and_bsmd_old_ambiguous_positional_form_now_raises():
    """The SPOD/BSMD reading of Analyzer(path, 256, 0.5) as nfft/overlap now raises."""
    with pytest.raises(TypeError, match=_POSITIONAL_TYPEERROR):
        SPODAnalyzer("data.mat", 256, 0.5)
    with pytest.raises(TypeError, match=_POSITIONAL_TYPEERROR):
        BSMDAnalyzer("data.mat", 256, 0.5)


def test_stpod_old_silent_positional_form_now_raises():
    """The historically silent STPODAnalyzer(path, 8, 10) call now raises."""
    with pytest.raises(TypeError, match=_POSITIONAL_TYPEERROR):
        STPODAnalyzer("data.mat", 8, 10)
