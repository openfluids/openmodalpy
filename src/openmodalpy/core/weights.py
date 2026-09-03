"""Spatial weights for the inner product each method uses.

A mode is only orthogonal under the weight the grid demands. These functions
build that weight for a uniform grid, a polar grid, or a grid of cell volumes,
and put it in the column shape the analyzers expect.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

try:
    from openmodalpy.core.parallel import (
        PARALLEL_AVAILABLE,
        calculate_polar_weights_optimized,
    )
except ImportError:
    PARALLEL_AVAILABLE = False


def _trapezoid_widths(a: np.ndarray, name: str) -> np.ndarray:
    """Trapezoid cell widths along one 1-D axis (half spacing at the ends).

    A single-point axis gets width 1.0 so an outer product across axes stays
    well defined (same convention as the polar helper).
    """
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D coordinate array to derive cell volumes; "
            f"got ndim={a.ndim}. Pass the axis coordinates, not a mesh."
        )
    if a.size == 0:
        raise ValueError(f"{name} is empty; cannot derive cell widths.")
    if a.size == 1:
        return np.ones(1)
    d = np.diff(a)
    if np.any(d <= 0.0):
        if np.all(d < 0.0):
            raise ValueError(
                f"{name} is strictly decreasing "
                f"({a[0]:g} -> {a[-1]:g}). Cell-volume weights require "
                f"strictly increasing coordinates; flip the axis first "
                f"(e.g. {name} = {name}[::-1] together with the matching "
                f"axis of q) instead of relying on a silent sort."
            )
        raise ValueError(
            f"{name} is not strictly increasing (first bad step near index "
            f"{int(np.argmax(d <= 0.0))}). Cell-volume weights refuse to "
            f"sort or repair coordinates; pass monotone 1-D {name}."
        )
    w = np.empty_like(a)
    w[0] = (a[1] - a[0]) / 2.0
    w[-1] = (a[-1] - a[-2]) / 2.0
    w[1:-1] = (a[2:] - a[:-2]) / 2.0
    return w


def _polar_theta_sector_fractions(z: np.ndarray) -> np.ndarray:
    """Sector fraction Delta-theta / (2*pi) for the polar third axis.

    ``z`` is azimuth theta in radians, strictly increasing, spanning at most
    one revolution (``theta_max - theta_min <= 2*pi + 1e-9``). Theta is
    periodic, so each point's width also counts the gap that wraps back to
    the first point one revolution later: append a virtual point at
    ``theta[0] + 2*pi``, take the plain trapezoid widths of that extended
    axis with ``_trapezoid_widths``, then fold the last (virtual) width back
    onto the first point. A partition covering the full circle then sums to
    exactly 2*pi, whatever its spacing.

    Two full-revolution samplings are accepted: half-open (``endpoint=False``
    style, the wrap gap back to ``theta[0] + 2*pi`` is close to the regular
    interior spacing) and closed (``z`` includes both 0 and ``2*pi``, wrap
    gap 0 -- the plain trapezoid widths of ``z`` itself already sum to
    ``2*pi`` and are used directly, with no wraparound extension). A theta
    axis covering only part of the circle -- a wrap gap much larger than the
    largest interior spacing -- is refused: sector weights assume one full
    revolution, and a wedge needs an explicit ``spatial_weights=`` metric.

    Flattened in the order documented in ``calculate_cell_volume_weights``.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 1:
        raise ValueError(
            "z must be a 1-D azimuth theta axis in radians; got ndim="
            f"{z_arr.ndim}. The polar third axis is azimuth, not a mesh."
        )
    if z_arr.size == 0:
        raise ValueError("z is empty; cannot derive azimuth sector widths.")
    if z_arr.size > 1 and np.any(np.diff(z_arr) <= 0.0):
        raise ValueError(
            f"z is not strictly increasing (range [{z_arr[0]:g}, {z_arr[-1]:g}] "
            "rad). The polar third axis is azimuth theta in radians; sort it "
            "before calling."
        )
    theta_range = float(z_arr[-1] - z_arr[0]) if z_arr.size > 1 else 0.0
    if theta_range > 2.0 * np.pi + 1e-9:
        raise ValueError(
            f"z spans {theta_range:g} rad, more than one revolution (2*pi "
            f"rad ~= {2.0 * np.pi:g}). The polar third axis is azimuth theta "
            "in radians, not a Cartesian z; a Cartesian z passed by mistake "
            "must be caught here."
        )
    if z_arr.size == 1:
        return np.array([1.0])
    wrap_gap = float((z_arr[0] + 2.0 * np.pi) - z_arr[-1])
    max_interior_gap = float(np.diff(z_arr).max())
    if wrap_gap > 1.5 * max_interior_gap:
        raise ValueError(
            f"z covers only part of one revolution: range [{z_arr[0]:g}, "
            f"{z_arr[-1]:g}] rad, wrap gap {wrap_gap:g} rad back to "
            f"theta[0] + 2*pi versus largest interior spacing "
            f"{max_interior_gap:g} rad. Sector weights assume a full "
            "revolution; a partial wedge needs an explicit spatial_weights= "
            "metric instead."
        )
    if abs(wrap_gap) < 1e-9:
        # Closed sampling: z already includes both 0 and 2*pi, so the plain
        # trapezoid widths already sum to 2*pi -- no wraparound extension.
        w = _trapezoid_widths(z_arr, "z")
        return w / (2.0 * np.pi)
    z_ext = np.concatenate([z_arr, [z_arr[0] + 2.0 * np.pi]])
    w_ext = _trapezoid_widths(z_ext, "z")
    w = w_ext[:-1].copy()
    w[0] += w_ext[-1]
    return w / (2.0 * np.pi)


def _coerce_spatial_weights(w: ArrayLike, expected_len: int) -> np.ndarray:
    """Accepted weight shapes -> 1-D vector of length ``expected_len``.

    Routes: 1-D; ``(n, 1)``; square matrix (its diagonal); non-square
    ``(n, k)`` row-major flatten; 3-D per-component stacked diagonals.

    A square (or 3-D stack of square planes) is always reduced to its
    diagonal: nothing off-diagonal is kept. The matrix is accepted (then
    reduced) only when it is numerically diagonal under the scale-invariant
    ratio

        r = max_{i != j} |W_ij| / sqrt(|W_ii| * |W_jj|)

    Reject when ``r > C * n * eps``, where ``n`` is the matrix side, ``eps``
    is machine epsilon of the array's own floating dtype (float64 for
    integer or object input), and ``C = 2``. At n = 4 that is a 3.8x margin
    over measured float32-after-arithmetic round-off (r ≈ 2.10 eps) and
    still rejects a float64 change of basis at 1e-14 (r ≈ 157 eps). A
    non-finite (NaN or inf) off-diagonal is rejected. A zero on the
    diagonal makes its row and column strict (any non-zero coupling gives
    r = inf). This package cannot represent off-diagonal coupling at all;
    pass ``np.diag(W)`` only if the diagonal is what was meant.
    """
    w = np.asarray(w)
    # C=2 → 3.8x margin over measured float32-after-arithmetic (r ≈ 2.10 eps).
    _diagonality_c = 2

    def _metric_eps(arr: np.ndarray) -> float:
        try:
            return float(np.finfo(arr.dtype).eps)
        except (TypeError, ValueError):
            return float(np.finfo(np.float64).eps)

    def _coupling_ratio(plane: np.ndarray) -> float:
        n = int(plane.shape[0])
        if n <= 1:
            return 0.0
        off_mask = ~np.eye(n, dtype=bool)
        off = plane[off_mask]
        abs_off = np.asarray(np.abs(off), dtype=np.float64)
        diag = np.asarray(np.abs(np.diag(plane)), dtype=np.float64)
        rows, cols = np.nonzero(off_mask)
        scale = np.sqrt(diag[rows] * diag[cols])
        ratio = np.zeros_like(abs_off)
        nz = scale > 0
        ratio[nz] = abs_off[nz] / scale[nz]
        ratio[~nz] = np.where(abs_off[~nz] > 0, np.inf, 0.0)
        return float(np.max(ratio))

    def _largest_rejected_offdiag(planes: list[np.ndarray]) -> float | None:
        # r > C * n * eps(dtype) per plane; non-finite off-diag => inf.
        worst: float | None = None
        for plane in planes:
            n = int(plane.shape[0])
            r = _coupling_ratio(plane)
            limit = _diagonality_c * n * _metric_eps(plane)
            if not np.isfinite(r) or r > limit:
                max_off = float(np.max(np.abs(plane - np.diag(np.diag(plane)))))
                worst = max_off if worst is None else max(worst, max_off)
        return worst

    def _reject_nondiagonal_square(max_off: float, shape: tuple[int, ...]) -> None:
        if len(shape) == 3:
            head = (
                "A 3-D stack of planes is read as stacked diagonals and cannot "
                f"represent off-diagonal coupling in an array of shape {shape}"
            )
        else:
            head = (
                "A square spatial metric is read as its diagonal and cannot "
                f"represent off-diagonal coupling in an array of shape {shape}"
            )
        raise ValueError(
            f"{head} (largest off-diagonal magnitude {max_off}). "
            "This package cannot represent off-diagonal coupling at all. "
            "If the diagonal is what was meant, pass np.diag(W)."
        )

    # Shape work only — do not cast to float yet. A complex array must reach
    # require_spatial_metric with its imaginary part intact; casting first would
    # emit ComplexWarning and hand the real part to the metric checks.
    if w.ndim == 3:
        if w.shape[0] != w.shape[1]:
            raise ValueError("weight array's first two dimensions must be equal")
        worst = _largest_rejected_offdiag([w[:, :, i] for i in range(w.shape[2])])
        if worst is not None:
            _reject_nondiagonal_square(worst, w.shape)
        w = np.stack([np.diag(w[:, :, i]) for i in range(w.shape[2])], axis=1)
    elif w.ndim == 2:
        if w.shape[0] == w.shape[1] and w.shape[1] != 1:
            worst = _largest_rejected_offdiag([w])
            if worst is not None:
                _reject_nondiagonal_square(worst, w.shape)
            w = np.diag(w)
        elif w.shape[1] > 1:
            w = w.reshape(-1)
        else:
            w = w.ravel()
    weights = np.asarray(w).reshape(-1)
    if weights.size != expected_len:
        raise ValueError(f"Weight vector length {weights.size} does not match n_space={expected_len}")
    if weights.dtype == object:
        # Same cast the return already performs. Doing it first lets
        # require_spatial_metric inspect a numeric object diagonal; np.isfinite
        # cannot read object dtype.
        weights = np.asarray(weights, dtype=float)
    require_spatial_metric(weights)
    return np.asarray(weights, dtype=float)


def _as_spatial_weight_column(w: ArrayLike, n_space: int | None = None) -> np.ndarray:
    """Accepted weight shapes -> column of shape ``(n_space, 1)``.

    ``n_space`` defaults to the length ``_coerce_spatial_weights`` would
    produce from ``w`` (vector length, column height, or square side).
    Pass it explicitly to reject a wrong-length input.
    """
    arr = np.asarray(w)
    if n_space is None:
        if arr.ndim == 3:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("weight array's first two dimensions must be equal")
            n_space = int(arr.shape[0] * arr.shape[2])
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[1] != 1:
            n_space = int(arr.shape[0])
        else:
            n_space = int(arr.size)
    return _coerce_spatial_weights(arr, int(n_space)).reshape(-1, 1)


def require_spatial_metric(weights: ArrayLike) -> None:
    """Raise ``ValueError`` if ``weights`` do not define an inner product.

    A metric that is not an inner product must not reach a solver. Isolated
    zeros among positive weights stay allowed -- a zero-measure cell
    contributes nothing (same as SPOD and BSMD). What is rejected: complex
    entries, non-finite entries, negative entries, and a zero total measure.
    Single definition, used by the decomposition seam, SPOD (via
    ``_coerce_spatial_weights``) and BSMD.
    """
    weights = np.asarray(weights)
    if np.iscomplexobj(weights):
        raise ValueError(
            "Spatial metric is complex. Casting it to real would silently discard the "
            "imaginary part and hand the solver a metric the caller never asked for."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            f"Spatial metric contains {np.count_nonzero(~np.isfinite(weights))} non-finite "
            "weight(s) (NaN or inf). This would otherwise surface much later as an "
            "unhelpful LAPACK error from inside the eigensolver."
        )
    if np.any(weights < 0):
        raise ValueError(
            f"Spatial metric contains {np.count_nonzero(weights < 0)} negative weight(s) "
            f"(most negative: {float(np.min(weights)):.6g}). A negative entry means the "
            "metric is not an inner product, so any energy computed from it is meaningless."
        )
    # Reached only when every weight is >= 0, so this means all of them are zero:
    # the domain has no measure, and the usual cause is worth naming.
    if np.sum(weights) <= 0:
        raise ValueError(
            f"Spatial metric has zero total measure ({weights.size} weights, all zero), so "
            "it defines no inner product. The usual cause is polar weights on a grid whose "
            "radial coordinate is 0: every annulus area is pi*r**2 = 0. Note the condition "
            "is r > 0, not Ny > 1 -- a single radial station at r > 0 is fine."
        )


def calculate_uniform_weights(
    x: np.ndarray, y: np.ndarray, z: ArrayLike | None = None, n_space: int | None = None
) -> np.ndarray:
    """Return uniform weights for a Cartesian grid or a scattered point set.

    Returns an all-ones column of length ``n_space`` when the coordinates are
    scattered (1-D ``x`` and ``y``, ``len(x) == len(y) == n_space``), and the
    tensor product ``Nx*Ny*Nz`` otherwise. The two readings collide only when
    ``n == 1``; scattered is preferred then only if ``n_space`` says so.
    With ``n_space=None`` the result is always the tensor product (historical
    behaviour). Grid spacing / cell volumes are not applied; callers that need
    a domain integral must supply their own W. Flattened in the order
    documented in ``calculate_cell_volume_weights``.
    """
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.ones((int(n_space), 1))
    # Support both 1-D and 2-D coordinate arrays
    if x.ndim > 1:
        Nx, Ny = x.shape
    elif y.ndim > 1:
        Nx, Ny = y.shape
    else:
        Nx, Ny = x.shape[0], y.shape[0]
    if z is None:
        Nz = 1
    else:
        z_arr = np.asarray(z)
        Nz = int(z_arr.shape[0] if z_arr.ndim > 0 else 1)
    return np.ones((Nx * Ny * Nz, 1))


def calculate_polar_weights(
    x: np.ndarray,
    y: np.ndarray,
    z: ArrayLike | None = None,
    use_parallel: bool = True,
    n_space: int | None = None,
) -> np.ndarray:
    """Calculate integration weights for a 2D cylindrical grid (x, r).

    With ``n_space`` set and ``x``/``y`` both 1-D of length ``n_space``, the
    coordinates are read as scattered points rather than grid axes: the
    weight per point is its radius, ``w_i = r_i = |y_i|``. This is the
    cylindrical Jacobian at the point, not a cell measure — it carries no
    integration cell, same as the scattered branch of
    ``calculate_uniform_weights``. ``z`` is ignored in the scattered branch.

    With ``z`` given (and not scattered), ``z`` is a 1-D azimuth axis theta in
    radians and the weight per (x, r, theta) cell is the 2-D (x, r) weight
    times the sector fraction ``Delta-theta / (2*pi)`` (see
    ``_polar_theta_sector_fractions``). Flattened in the order documented in
    ``calculate_cell_volume_weights``.
    """
    z_arr = None if z is None else np.asarray(z, dtype=np.float64)
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.abs(y).reshape(int(n_space), 1)
    if use_parallel and PARALLEL_AVAILABLE:
        return calculate_polar_weights_optimized(x, y, z=z_arr, n_space=n_space)
    # Support both 1-D and 2-D coordinate arrays
    x_line = x[:, 0] if x.ndim > 1 else x
    y_line = y[0, :] if y.ndim > 1 else y
    Nx = x_line.shape[0]
    Ny = y_line.shape[0]

    # Calculate y-direction (r-direction) integration weights (Wy)
    Wy = np.zeros((Ny, 1))

    # First point (centerline)
    if Ny > 1:
        y_mid_right = (y_line[0] + y_line[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y_line[0] ** 2

    # Middle points
    for i in range(1, Ny - 1):
        y_mid_left = (y_line[i - 1] + y_line[i]) / 2
        y_mid_right = (y_line[i] + y_line[i + 1]) / 2
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)

    # Last point
    if Ny > 1:
        y_mid_left = (y_line[-2] + y_line[-1]) / 2
        Wy[Ny - 1] = np.pi * (y_line[-1] ** 2 - y_mid_left**2)

    # Calculate x-direction integration weights (Wx)
    Wx = np.zeros((Nx, 1))

    # First point
    if Nx > 1:
        Wx[0] = (x_line[1] - x_line[0]) / 2
    else:
        Wx[0] = 1.0

    # Middle points
    for i in range(1, Nx - 1):
        Wx[i] = (x_line[i + 1] - x_line[i - 1]) / 2

    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x_line[Nx - 1] - x_line[Nx - 2]) / 2

    if z_arr is None:
        # Combine weights: (Ny, Nx) outer product, flattened C-order.
        W = np.reshape(np.outer(Wy.ravel(), Wx.ravel()), (Nx * Ny, 1))
        return W

    # 3-D polar: fold in the azimuth sector fraction and flatten (theta, r, x).
    theta_fraction = _polar_theta_sector_fractions(z_arr)
    volumes_2d = np.outer(Wy.ravel(), Wx.ravel())  # (Ny, Nx)
    volumes = theta_fraction[:, None, None] * volumes_2d[None, :, :]  # (Ntheta, Ny, Nx)
    return volumes.reshape(-1, 1)


def calculate_cell_volume_weights(x: ArrayLike, y: ArrayLike, z: ArrayLike | None = None) -> np.ndarray:
    """Cell-volume weights for a (possibly stretched) Cartesian grid.

    Each axis contributes trapezoid cell widths (half the neighbouring
    spacing at the boundary points); the outer product across x, y[, z] gives
    one weight per cell, flattened in C-order to match the snapshot layout
    ``(Ns, Ny*Nx*Nz)``: ``index = ((iz*Ny + iy)*Nx + ix)`` (``iy*Nx + ix``
    in 2-D). Coordinates must be strictly increasing 1-D arrays; anything
    else raises ``ValueError`` — decreasing axes are told to flip, never
    sorted silently.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    wx = _trapezoid_widths(x_arr, "x")
    wy = _trapezoid_widths(y_arr, "y")
    volumes = np.outer(wy, wx)  # (Ny, Nx); C-order flatten -> iy*Nx + ix
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64)
        wz = _trapezoid_widths(z_arr, "z")
        volumes = wz[:, None, None] * volumes[None, :, :]  # (Nz, Ny, Nx)
    return volumes.reshape(-1, 1)
