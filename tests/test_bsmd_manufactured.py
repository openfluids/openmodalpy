"""Manufactured-solution oracle for the BSMD triad-correlation matrix.

This test does NOT assume where the conjugation belongs in
``_compute_single_triad`` (src/openmodalpy/bsmd.py:552-553). It builds a
synthetic field with a phase-locked quadratic triad from first principles
(known analytically to have a non-zero bispectrum) and a matched "control"
field with the same spectral content but an independent, non-phase-locked
third component (known analytically to have a vanishing bispectrum in the
block-average limit). It then checks whether the BSMD implementation
actually discriminates between the two.

Honesty note on which tests actually discriminate between the conjugated and
unconjugated forms of the cross-bispectral matrix:
  - test_cubic_amplitude_scaling passes under BOTH forms. Cubic scaling in
    amplitude is a correctness invariant of the block-averaged product, not a
    property specific to conjugation, so it is NOT a discriminator.
  - test_seed_invariance_of_resonant_eigenvalue, test_resonant_eigenvalue_is_real,
    and the locked-vs-control comparisons ARE the discriminating tests: the true
    bispectrum lambda = alpha^3 * A * B * C * sum_x(w * f1 * f2 * f3) has no phase
    dependence at all (the block phases cancel exactly), so only the correctly
    conjugated form is seed-invariant and real. The unconjugated form gives
    (g/N) * sum_b exp(2i*theta_b), a phase random walk that is seed-dependent and
    generically complex.
"""

import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer

# ---------------------------------------------------------------------------
# Manufactured-field construction
# ---------------------------------------------------------------------------

NFFT = 32
OVERLAP = 0.0
NBLOCKS = 64
NS = NBLOCKS * NFFT  # 2048, exact (novlap=0) so nblocks comes out to exactly 64
NX, NY = 4, 4
NSPACE = NX * NY
DT = 1.0

K1, K2, K3 = 3, 5, 8  # k1 + k2 == k3
K4 = K3 + K1  # = 11; independent (non-locked) fourth component
A, B, C = 1.0, 0.8, 0.6
D = 0.5

RESONANT_TRIADS = [(3, 5, 8), (8, -5, 3)]
# Bins referenced by these triads contain NO energy at all in the manufactured
# field, so their |lambda| is trivially ~0 (floating-point noise) regardless of
# conjugation. They give the "dominance" picture but are not a strong control.
NON_RESONANT_TRIADS = [(2, 5, 7), (1, 5, 6), (4, 5, 9), (2, 3, 5), (3, 6, 9), (7, -5, 2)]
# (3, 8, 11) / (11, -8, 3): all three bins ARE populated (via the K4 component),
# but K4's phase is independent of th1/th3 in BOTH the locked and control field,
# so this triad is NOT phase-locked. Unlike the triads above, an implementation
# cannot pass this control "for free" by simply having zero energy at those
# bins -- this is the genuine, non-gameable non-resonant control.
NON_RESONANT_POPULATED_TRIADS = [(3, 8, 11), (11, -8, 3)]
ALL_NON_RESONANT_TRIADS = NON_RESONANT_TRIADS + NON_RESONANT_POPULATED_TRIADS
ALL_TRIADS = RESONANT_TRIADS + ALL_NON_RESONANT_TRIADS


def _spatial_patterns(seed=12345):
    """Four fixed, distinct, real, unit-norm spatial patterns of length NSPACE."""
    rng = np.random.default_rng(seed)
    f1, f2, f3, f4 = rng.standard_normal((4, NSPACE))
    f1 /= np.linalg.norm(f1)
    f2 /= np.linalg.norm(f2)
    f3 /= np.linalg.norm(f3)
    f4 /= np.linalg.norm(f4)
    return f1, f2, f3, f4


def _make_field(locked: bool, seed=987):
    """Build the (NS, NSPACE) manufactured field.

    locked=True:  th3 = th1 + th2 (phase-locked quadratic triad).
    locked=False: th3 independent of th1, th2 (control, no true bispectrum).

    In both cases, a fourth component at bin K4 = K3 + K1 = 11 carries its own
    independent random phase th4 (never locked to th1/th3), so that the triad
    (3, 8, 11) has energy in all three bins but is genuinely not phase-locked.
    """
    f1, f2, f3, f4 = _spatial_patterns()
    rng = np.random.default_rng(seed)

    q = np.zeros((NS, NSPACE))
    t_local = np.arange(NFFT)  # local-within-block time index, 0..nfft-1

    for b in range(NBLOCKS):
        th1, th2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
        if locked:
            th3 = th1 + th2
        else:
            th3 = rng.uniform(0.0, 2.0 * np.pi)
        th4 = rng.uniform(0.0, 2.0 * np.pi)  # independent, always non-locked

        c1 = A * np.cos(2.0 * np.pi * K1 * t_local / NFFT + th1)
        c2 = B * np.cos(2.0 * np.pi * K2 * t_local / NFFT + th2)
        c3 = C * np.cos(2.0 * np.pi * K3 * t_local / NFFT + th3)
        c4 = D * np.cos(2.0 * np.pi * K4 * t_local / NFFT + th4)

        block = np.outer(c1, f1) + np.outer(c2, f2) + np.outer(c3, f3) + np.outer(c4, f4)
        q[b * NFFT : (b + 1) * NFFT, :] = block

    return q


def _make_analyzer(tmp_path, q, triads, window_type="boxcar", tag="dummy"):
    # NB: BSMDAnalyzer's FFT-block cache is keyed only on (data_root, nfft,
    # overlap, Ns) — NOT on the actual q content — so distinct calls sharing
    # a results_dir/file_path basename would silently reuse a stale cached
    # qhat from a *different* q array. Give each analyzer instance its own
    # subdirectory and file basename (`tag`) so every call gets a fresh cache
    # file and this test genuinely recomputes the FFT each time.
    out_dir = tmp_path / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    Nx, Ny = NX, NY
    data = {
        "q": q,
        "x": np.linspace(0, 1, Nx),
        "y": np.linspace(0, 1, Ny),
        "dt": DT,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": q.shape[0],
    }
    analyzer = BSMDAnalyzer(
        file_path=f"{tag}.h5",
        nfft=NFFT,
        overlap=OVERLAP,
        results_dir=out_dir,
        figures_dir=out_dir,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=triads,
    )
    # BSMDAnalyzer's constructor has no window_type parameter; the base class
    # reads self.window_type via getattr(..., "hamming") default in
    # compute_fft_blocks(). We force a rectangular (boxcar) window here so the
    # manufactured tones sit exactly on FFT bins with zero spectral leakage.
    analyzer.window_type = window_type
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    assert analyzer.nblocks == NBLOCKS, (
        f"Expected {NBLOCKS} blocks from Ns={NS}, nfft={NFFT}, novlap=0; got {analyzer.nblocks}"
    )
    return analyzer


def _eigenvalue_table(analyzer):
    """Map triad tuple -> |eigenvalue| from a completed static BSMD run."""
    out = {}
    for triad, lam in zip(analyzer.triads, analyzer.eigenvalues):
        out[tuple(int(x) for x in triad)] = abs(lam)
    return out


def _print_table(title, table):
    print(f"\n=== {title} ===")
    for triad in table:
        tag = "RESONANT" if triad in RESONANT_TRIADS else "non-resonant"
        print(f"  triad={triad!s:14s} |lambda|={table[triad]:.6e}  [{tag}]")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resonant_triad_dominates(tmp_path):
    q_locked = _make_field(locked=True, seed=1)
    analyzer = _make_analyzer(tmp_path, q_locked, ALL_TRIADS, tag="locked")
    analyzer._perform_static_bsmd_core()
    table = _eigenvalue_table(analyzer)
    _print_table("LOCKED FIELD: eigenvalue table", table)

    # Triads whose bins carry no energy at all: trivial, ~0 floor. The
    # resonant triad must dominate these by a wide (100x) margin.
    max_non_resonant = max(table[t] for t in NON_RESONANT_TRIADS)
    print(f"max |lambda| (non-resonant, no energy) = {max_non_resonant:.6e}")
    for triad in RESONANT_TRIADS:
        ratio = table[triad] / max_non_resonant
        print(f"triad {triad}: |lambda|={table[triad]:.6e}, ratio to max non-resonant = {ratio:.3e}")
        assert table[triad] >= 100.0 * max_non_resonant, (
            f"Resonant triad {triad} did not dominate: |lambda|={table[triad]:.3e} vs "
            f"100x max non-resonant={100 * max_non_resonant:.3e}"
        )

    # Populated-but-unlocked triads (3, 8, 11) / (11, -8, 3): all three bins
    # have energy, but K4's phase is independent, so the true bispectrum is
    # zero and the finite-block estimate should sit at the statistical floor
    # ~ 1/sqrt(N_blk) = 1/sqrt(64) = 1/8 of the resonant value. We only assert
    # the conservative 4x bound here (NOT 100x) -- 100x would be dishonest
    # given the honest sqrt(N_blk) floor.
    for triad in NON_RESONANT_POPULATED_TRIADS:
        for resonant in RESONANT_TRIADS:
            ratio = table[resonant] / table[triad]
            print(
                f"resonant {resonant} vs populated-unlocked {triad}: |lambda|={table[triad]:.6e}, ratio = {ratio:.3e}"
            )
            assert table[resonant] >= 4.0 * table[triad], (
                f"Resonant triad {resonant} did not clear the statistical floor of "
                f"populated-but-unlocked triad {triad}: |lambda|={table[resonant]:.3e} vs "
                f"4x floor={4.0 * table[triad]:.3e}"
            )


def test_control_field_has_no_triad_energy(tmp_path):
    # The control holds spectral content fixed and varies ONLY phase coherence:
    # bins (3, 5, 8) carry the same energy as in the locked field, but th3 is
    # independent of th1+th2. So the comparison that matters is the SAME
    # resonant triad in both fields -- comparing against the genuinely-empty
    # bins would only re-test test_resonant_triad_dominates.
    #
    # Bound: the unlocked estimate is a 1/sqrt(N_blk) random walk, so the
    # expected separation is sqrt(64) = 8. 4x is the conservative floor. A /100
    # bound here would be physically unjustified and would fail for the right
    # implementation.
    triads = RESONANT_TRIADS + NON_RESONANT_TRIADS
    q_locked = _make_field(locked=True, seed=1)
    q_control = _make_field(locked=False, seed=2)

    analyzer_locked = _make_analyzer(tmp_path, q_locked, triads, tag="locked")
    analyzer_locked._perform_static_bsmd_core()
    table_locked = _eigenvalue_table(analyzer_locked)

    analyzer_control = _make_analyzer(tmp_path, q_control, triads, tag="control")
    analyzer_control._perform_static_bsmd_core()
    table_control = _eigenvalue_table(analyzer_control)
    _print_table("CONTROL FIELD (unlocked phase): eigenvalue table", table_control)

    for triad in RESONANT_TRIADS:
        lam_locked = table_locked[triad]
        lam_control = table_control[triad]
        ratio = lam_locked / lam_control
        print(
            f"triad {triad}: locked |lambda|={lam_locked:.6e}  "
            f"control |lambda|={lam_control:.6e}  locked/control={ratio:.3f}"
        )
        assert lam_locked >= 4.0 * lam_control, (
            f"Breaking the phase lock did not collapse triad {triad}: "
            f"locked |lambda|={lam_locked:.3e} is not >= 4x control "
            f"|lambda|={lam_control:.3e} (ratio {ratio:.3f}). The estimator does "
            "not distinguish genuine quadratic phase coupling from decorrelated "
            "energy at the same frequencies."
        )


def test_seed_invariance_of_resonant_eigenvalue(tmp_path):
    """THE decisive discriminator: the resonant eigenvalue must not depend on the
    RNG phase seed at all.

    For the true bispectrum, lambda = alpha^3 * A * B * C * sum_x(w * f1 * f2 * f3),
    which contains no phase dependence whatsoever -- the block phases th1, th2, th3
    cancel exactly under the correct (conjugated) sum-frequency convention. An
    unconjugated form instead gives (g/N_blk) * sum_b exp(2i*theta_b), a random walk
    over the per-block phase realization, so it would vary with the seed.
    """
    triad = (3, 5, 8)
    eigenvalues = {}
    for seed in (1, 7, 99):
        q_locked = _make_field(locked=True, seed=seed)
        analyzer = _make_analyzer(tmp_path, q_locked, [triad], tag=f"seed_{seed}")
        analyzer._perform_static_bsmd_core()
        eigenvalues[seed] = analyzer.eigenvalues[0]
        print(f"seed={seed}: lambda={analyzer.eigenvalues[0]!r}")

    lam_ref = eigenvalues[1]
    for seed in (7, 99):
        np.testing.assert_allclose(
            eigenvalues[seed],
            lam_ref,
            rtol=1e-10,
            err_msg=f"Resonant eigenvalue not seed-invariant: seed={seed} lambda={eigenvalues[seed]!r} "
            f"vs seed=1 lambda={lam_ref!r}",
        )


def test_resonant_eigenvalue_is_real(tmp_path):
    """The resonant eigenvalue must be purely real to machine precision.

    sum_x(w * f1 * f2 * f3) is a sum of real quantities, so lambda has zero
    imaginary part analytically; a non-negligible imaginary part indicates the
    sum-frequency term is not correctly conjugated.
    """
    triad = (3, 5, 8)
    q_locked = _make_field(locked=True, seed=1)
    analyzer = _make_analyzer(tmp_path, q_locked, [triad], tag="realness")
    analyzer._perform_static_bsmd_core()
    lam = analyzer.eigenvalues[0]
    rel_imag = abs(lam.imag) / abs(lam)
    print(f"lambda={lam!r}, |Im lambda|/|lambda|={rel_imag:.3e}")
    assert rel_imag < 1e-12, f"Resonant eigenvalue is not real: |Im lambda|/|lambda|={rel_imag:.3e} >= 1e-12"


@pytest.mark.parametrize("alpha", [2.0, 3.0])
def test_cubic_amplitude_scaling(tmp_path, alpha):
    q_locked = _make_field(locked=True, seed=1)
    triad = (3, 5, 8)

    analyzer_base = _make_analyzer(tmp_path, q_locked, [triad], tag="base")
    analyzer_base._perform_static_bsmd_core()
    lam_base = analyzer_base.eigenvalues[0]

    analyzer_scaled = _make_analyzer(tmp_path, alpha * q_locked, [triad], tag=f"scaled_{alpha}")
    analyzer_scaled._perform_static_bsmd_core()
    lam_scaled = analyzer_scaled.eigenvalues[0]

    predicted = (alpha**3) * lam_base
    print(f"alpha={alpha}: lam_base={lam_base}, lam_scaled={lam_scaled}, predicted={predicted}")
    np.testing.assert_allclose(lam_scaled, predicted, rtol=1e-8)
