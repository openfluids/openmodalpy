"""Plotting functions for spatial and volumetric modal decomposition results.

Render modes as contour, slice, and isometric plots with optional styling metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from openmodalpy.core.config import CMAP_DIV

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def get_fig_aspect_ratio(data: dict, clamp_low: float = 0.3, clamp_high: float = 5.0) -> float:
    """Return physical domain aspect ratio (dx/dy) with reasonable clamping for figure sizing.

    Computes aspect from physical extent if coordinates available, otherwise from Nx/Ny.
    Clamps to [0.3, 5.0] to avoid extremely distorted figures while preserving physical proportions.
    """
    x_coords = data.get("x")
    y_coords = data.get("y")

    # Try to compute from physical extent first
    if x_coords is not None and y_coords is not None:
        try:
            x_arr = np.asarray(x_coords)
            y_arr = np.asarray(y_coords)
            dx = float(x_arr.max() - x_arr.min())
            dy = float(y_arr.max() - y_arr.min())
            if dx > 0 and dy > 0:
                aspect = dx / dy
                return max(clamp_low, min(aspect, clamp_high))
        except (ValueError, TypeError):
            pass

    # Fall back to grid point ratio
    nx = int(data.get("Nx", 1))
    ny = int(data.get("Ny", 1))
    if ny <= 0:
        aspect = 1.0
    else:
        aspect = nx / ny
    return max(clamp_low, min(aspect, clamp_high))


def get_plot_style(data: dict, section: str = "spatial") -> dict[str, Any]:
    """Return plot-style overrides stored in data metadata."""
    metadata = data.get("metadata", {})
    plot_style = metadata.get("plot_style", {})
    if not isinstance(plot_style, dict):
        return {}
    section_style = plot_style.get(section)
    if isinstance(section_style, dict):
        return section_style
    return plot_style


def style_spatial_axes(
    ax: Axes,
    data: dict,
    *,
    x_coords: ArrayLike | None = None,
    y_coords: ArrayLike | None = None,
    equal_default: bool = True,
) -> None:
    """Apply metadata-driven styling to a 2D spatial axis."""
    style = get_plot_style(data)
    figure_facecolor = style.get("figure_facecolor")
    axes_facecolor = style.get("axes_facecolor", style.get("facecolor"))
    if figure_facecolor:
        ax.figure.patch.set_facecolor(figure_facecolor)
    if axes_facecolor:
        ax.set_facecolor(axes_facecolor)

    axis_labels = style.get("axis_labels", {})
    ax.set_xlabel(axis_labels.get("x", r"$x/D$"))
    ax.set_ylabel(axis_labels.get("y", r"$y/D$"))

    aspect = style.get("aspect")
    if aspect == "equal":
        ax.set_aspect("equal", "box")
    elif aspect == "auto":
        ax.set_aspect("auto")
    elif aspect is not None:
        ax.set_aspect(aspect)
    elif equal_default:
        ax.set_aspect("equal", "box")
    else:
        ax.set_aspect("auto")

    if x_coords is not None:
        x_arr = np.asarray(x_coords)
        x_limits = style.get("xlim", [float(np.min(x_arr)), float(np.max(x_arr))])
        ax.set_xlim(*x_limits)
    elif "xlim" in style:
        ax.set_xlim(*style["xlim"])

    if y_coords is not None:
        y_arr = np.asarray(y_coords)
        y_limits = style.get("ylim", [float(np.min(y_arr)), float(np.max(y_arr))])
        ax.set_ylim(*y_limits)
    elif "ylim" in style:
        ax.set_ylim(*style["ylim"])

    grid_style = style.get("grid", {})
    if isinstance(grid_style, dict):
        grid_enabled = grid_style.get("enabled", True)
        grid_kwargs = {
            "linestyle": grid_style.get("linestyle", "--"),
            "alpha": grid_style.get("alpha", 0.3),
            "color": grid_style.get("color"),
        }
    else:
        grid_enabled = bool(grid_style) if style.get("grid") is not None else True
        grid_kwargs = {"linestyle": "--", "alpha": 0.3, "color": None}
    if grid_enabled:
        if grid_kwargs["color"] is None:
            del grid_kwargs["color"]
        ax.grid(True, **grid_kwargs)
    else:
        ax.grid(False)


def add_inset_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
    data: dict,
    *,
    ticks: Sequence[float] | None = None,
    ticklabels: Sequence[str] | None = None,
    fmt: str = "%.2f",
) -> Colorbar | None:
    """Add a compact, metadata-driven inset colorbar to an axis."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    style = get_plot_style(data)
    cbar_style = style.get("colorbar", {})
    if cbar_style.get("enabled", True) is False:
        return None

    location = cbar_style.get("location", "top_inset")
    orientation = cbar_style.get("orientation", "horizontal")
    if location == "top_inset":
        cax = inset_axes(
            ax,
            width=cbar_style.get("width", "24%"),
            height=cbar_style.get("height", "6%"),
            loc=cbar_style.get("loc", "upper right"),
            borderpad=cbar_style.get("borderpad", 2.0),
        )
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation, format=fmt)
    else:
        cb = fig.colorbar(mappable, ax=ax, shrink=cbar_style.get("shrink", 0.8), format=fmt)
        cax = cb.ax

    cb.ax.tick_params(
        labelsize=cbar_style.get("tick_fontsize", 8),
        pad=cbar_style.get("tick_pad", 1),
        colors=cbar_style.get("tick_color", "black"),
    )
    if orientation == "horizontal":
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cax.patch.set_facecolor(cbar_style.get("facecolor", "white"))
    cax.patch.set_alpha(cbar_style.get("alpha", 0.95))
    return cb


def plot_orthogonal_slices_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    output_path: str,
    title_prefix: str,
    data: dict,
    slice_indices: tuple[int, int, int] | None = None,
    scalar_name: str = "mode",
) -> None:
    """Render 3 orthogonal slices of a 3D scalar field with PyVista."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyVista is required for 3D slice plots. Install openmodalpy[viz3d] to enable 3D plotting."
        ) from exc

    values, x_arr, y_arr, z_arr = subset_volume_focus_3d(field_3d, x_coords, y_coords, z_coords, data)
    nx, ny, nz = values.shape

    if slice_indices is None:
        slice_indices = (nx // 2, ny // 2, nz // 2)
    ix, iy, iz = slice_indices

    vmin, vmax = get_robust_clim(values, method="percentile")
    center = [float(x_arr[ix]), float(y_arr[iy]), float(z_arr[iz])]

    grid = pv.RectilinearGrid(x_arr, y_arr, z_arr)
    grid.point_data[scalar_name] = values.flatten(order="F")

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600), border=False)
    plotter.set_background("white")

    slice_specs = [
        ("YZ", "x", [center[0], center[1], center[2]], plotter.view_yz, f"x = {center[0]:.3g}"),
        ("XZ", "y", [center[0], center[1], center[2]], plotter.view_xz, f"y = {center[1]:.3g}"),
        ("XY", "z", [center[0], center[1], center[2]], plotter.view_xy, f"z = {center[2]:.3g}"),
    ]

    for idx, (plane_name, normal, origin, view_fn, coord_label) in enumerate(slice_specs):
        plotter.subplot(0, idx)
        slc = grid.slice(normal=normal, origin=origin)
        plotter.add_mesh(
            slc,
            scalars=scalar_name,
            cmap=CMAP_DIV,
            clim=[vmin, vmax],
            show_scalar_bar=(idx == 2),
            scalar_bar_args={"title": "", "n_labels": 3},
        )
        plotter.add_text(f"{title_prefix}\n{plane_name} @ {coord_label}", font_size=10)
        view_fn()
        plotter.enable_parallel_projection()
        plotter.show_bounds(
            grid="front",
            location="outer",
            ticks="outside",
            xtitle="x",
            ytitle="y",
            ztitle="z",
            font_size=9,
            minor_ticks=False,
        )

    plotter.screenshot(output_path)
    plotter.close()
    logger.info("Saving figure %s", output_path)


def plot_isometric_slices_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    output_path: str,
    title_prefix: str,
    data: dict,
    slice_indices: tuple[int, int, int] | None = None,
    scalar_name: str = "mode",
) -> None:
    """Render positive/negative 3D isosurfaces in one isometric PyVista view."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyVista is required for 3D isometric plots. Install openmodalpy[viz3d] to enable 3D plotting."
        ) from exc

    values, x_arr, y_arr, z_arr = subset_volume_focus_3d(field_3d, x_coords, y_coords, z_coords, data)
    nx, ny, nz = values.shape

    vmin, vmax = get_robust_clim(values, method="percentile")
    grid = pv.RectilinearGrid(x_arr, y_arr, z_arr)
    grid.point_data[scalar_name] = values.flatten(order="F")
    abs_scale = max(abs(vmin), abs(vmax))
    if abs_scale <= 0:
        raise ValueError("Cannot build isosurfaces from a zero field.")
    iso_value = 0.45 * abs_scale

    positive = grid.contour(isosurfaces=[iso_value], scalars=scalar_name)
    negative = grid.contour(isosurfaces=[-iso_value], scalars=scalar_name)

    bounds_sources = [mesh.bounds for mesh in (positive, negative) if mesh.n_points]
    if bounds_sources:
        xmin = min(bounds[0] for bounds in bounds_sources)
        xmax = max(bounds[1] for bounds in bounds_sources)
        ymin = min(bounds[2] for bounds in bounds_sources)
        ymax = max(bounds[3] for bounds in bounds_sources)
        zmin = min(bounds[4] for bounds in bounds_sources)
        zmax = max(bounds[5] for bounds in bounds_sources)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    span_z = max(zmax - zmin, 1e-12)
    shock_xmax = xmin + 0.40 * span_x
    focus_bounds = (
        xmin,
        shock_xmax,
        ymin - 0.12 * span_y,
        ymax + 0.12 * span_y,
        zmin - 0.12 * span_z,
        zmax + 0.12 * span_z,
    )
    positive = positive.clip_box(bounds=focus_bounds, invert=False)
    negative = negative.clip_box(bounds=focus_bounds, invert=False)

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900), border=False)
    plotter.set_background("white")
    if positive.n_points:
        plotter.add_mesh(
            positive,
            color="#d1495b",
            opacity=0.62,
            smooth_shading=True,
            specular=0.2,
        )
    if negative.n_points:
        plotter.add_mesh(
            negative,
            color="#3a86ff",
            opacity=0.62,
            smooth_shading=True,
            specular=0.2,
        )

    bounds_sources = [mesh.bounds for mesh in (positive, negative) if mesh.n_points]
    if bounds_sources:
        xmin = min(bounds[0] for bounds in bounds_sources)
        xmax = max(bounds[1] for bounds in bounds_sources)
        ymin = min(bounds[2] for bounds in bounds_sources)
        ymax = max(bounds[3] for bounds in bounds_sources)
        zmin = min(bounds[4] for bounds in bounds_sources)
        zmax = max(bounds[5] for bounds in bounds_sources)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    span_z = max(zmax - zmin, 1e-12)
    pad_x = 0.12 * span_x
    pad_y = 0.12 * span_y
    pad_z = 0.12 * span_z
    focus_bounds = (
        xmin - pad_x,
        xmax + pad_x,
        ymin - pad_y,
        ymax + pad_y,
        zmin - pad_z,
        zmax + pad_z,
    )
    focus_center = (
        0.5 * (focus_bounds[0] + focus_bounds[1]),
        0.5 * (focus_bounds[2] + focus_bounds[3]),
        0.5 * (focus_bounds[4] + focus_bounds[5]),
    )
    max_span = max(
        focus_bounds[1] - focus_bounds[0],
        focus_bounds[3] - focus_bounds[2],
        focus_bounds[5] - focus_bounds[4],
    )

    plotter.add_mesh(
        pv.Box(bounds=focus_bounds),
        style="wireframe",
        color="black",
        line_width=1,
        opacity=0.35,
    )
    plotter.add_text(f"{title_prefix}\niso = ±{iso_value:.3g}", font_size=11)
    plotter.camera.focal_point = focus_center
    plotter.camera.position = (
        focus_center[0] + 1.45 * max_span,
        focus_center[1] + 0.55 * max_span,
        focus_center[2] - 1.35 * max_span,
    )
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.clipping_range = (1e-3, 50.0 * max_span)
    plotter.screenshot(output_path)
    plotter.close()
    logger.info("Saving figure %s", output_path)


def plot_modes_3d(
    kind: str,
    work_items: Iterable[Mapping[str, Any]],
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    data: dict,
) -> None:
    """Dispatch a sequence of 3D mode plots to the slices or isometric renderer.

    Each work item is a mapping with required keys ``mode_3d``, ``output_path``,
    and ``title_prefix``. Optional ``scalar_name`` is forwarded when present so
    callers that omit it keep the renderer default.
    """
    if kind == "slices":
        plot_fn = plot_orthogonal_slices_3d
    elif kind == "isometric":
        plot_fn = plot_isometric_slices_3d
    else:
        raise ValueError(f"kind must be 'slices' or 'isometric', got {kind!r}")

    for item in work_items:
        kwargs = {
            "output_path": item["output_path"],
            "title_prefix": item["title_prefix"],
            "data": data,
        }
        if "scalar_name" in item:
            kwargs["scalar_name"] = item["scalar_name"]
        plot_fn(item["mode_3d"], x_coords, y_coords, z_coords, **kwargs)


def get_robust_clim(data: np.ndarray, method: str = "percentile", sigma: float = 2.5) -> tuple:
    """Compute robust colormap limits that reduce the effect of outliers.

    Parameters
    ----------
    data : ndarray
        Data array (can contain NaNs which will be ignored)
    method : str
        'percentile' : Use 2nd and 98th percentiles
        'sigma' : Use median ± sigma * MAD (median absolute deviation)
        'minmax' : Use global min/max (no robustness)
    sigma : float
        Number of standard deviations for 'sigma' method

    Returns
    -------
    vmin, vmax : float
        Colormap limits
    """
    arr = np.asarray(data)
    if arr.size == 0:
        return -1.0, 1.0
    # Fancy-indexing with isfinite always copies, even when every value is finite.
    # Mode volumes are finite; keep that path allocation-light. Still drop NaN and
    # +/-Inf when present (np.nanpercentile keeps Inf, so it is not a substitute).
    if np.isfinite(arr).all():
        data_clean = arr
    else:
        flat = arr.ravel()
        data_clean = flat[np.isfinite(flat)]
        if data_clean.size == 0:
            return -1.0, 1.0

    if method == "percentile":
        vmin, vmax = np.percentile(data_clean, [2, 98])
    elif method == "sigma":
        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        # MAD to std: std ≈ 1.4826 * MAD
        std_estimate = 1.4826 * mad
        vmin = median - sigma * std_estimate
        vmax = median + sigma * std_estimate
    else:  # minmax
        vmin, vmax = data_clean.min(), data_clean.max()

    # Ensure symmetric for diverging colormaps
    abs_max = max(abs(vmin), abs(vmax))
    if not np.isfinite(abs_max) or abs_max == 0.0:
        return -1.0, 1.0
    return -abs_max, abs_max


def format_mode_title(data: dict, mode_index: int, default: str) -> str:
    """Format a mode title using optional metadata-driven templates."""
    style = get_plot_style(data)
    template = style.get("title_template")
    if not template:
        return default
    mode_number = mode_index + 1
    return template.format(mode=mode_number, m=mode_number)


def subset_volume_focus_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply optional metadata-driven volume cropping to a 3D scalar field."""
    values = np.asarray(field_3d)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D field, got shape {values.shape}.")

    x_arr = np.asarray(x_coords)
    y_arr = np.asarray(y_coords)
    z_arr = np.asarray(z_coords)
    nx, ny, nz = values.shape
    if x_arr.shape[0] != nx or y_arr.shape[0] != ny or z_arr.shape[0] != nz:
        raise ValueError(
            f"Coordinate lengths {(x_arr.shape[0], y_arr.shape[0], z_arr.shape[0])} do not match field shape {values.shape}."
        )

    style = get_plot_style(data, section="volume")

    def _axis_subset(arr: np.ndarray, limits: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            mask = np.ones(arr.shape[0], dtype=bool)
            return arr, mask
        lo = max(float(np.min(arr)), float(limits[0]))
        hi = min(float(np.max(arr)), float(limits[1]))
        mask = (arr >= lo) & (arr <= hi)
        if not np.any(mask):
            raise ValueError(f"Requested volume limits {limits} do not overlap the available coordinate range.")
        return arr[mask], mask

    x_focus, x_mask = _axis_subset(x_arr, style.get("xlim"))
    y_focus, y_mask = _axis_subset(y_arr, style.get("ylim"))
    z_focus, z_mask = _axis_subset(z_arr, style.get("zlim"))
    # np.ix_ fancy indexing always copies. When no axis is cropped (the default),
    # return the input array as a view so the uncropped path pays no volume copy.
    if x_mask.all() and y_mask.all() and z_mask.all():
        focused = values
    else:
        focused = values[np.ix_(x_mask, y_mask, z_mask)]
    return focused, x_focus, y_focus, z_focus
