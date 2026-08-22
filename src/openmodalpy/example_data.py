"""Built-in synthetic datasets used by the example configs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def generate_double_gyre(
    Nx: int = 80,
    Ny: int = 40,
    Nt: int = 200,
    A: float = 0.25,
    epsilon: float = 0.25,
    period: float = 10.0,
    t_max: float = 20.0,
) -> dict[str, Any]:
    """Generate a double-gyre velocity field and return the ``u`` snapshots."""
    omega = 2 * np.pi / period

    x = np.linspace(0.0, 2.0, Nx)
    y = np.linspace(0.0, 1.0, Ny)
    t = np.linspace(0.0, t_max, Nt)
    dt = float(t[1] - t[0])
    X, Y = np.meshgrid(x, y)  # contract layout: (Ny, Nx), index = iy*Nx + ix

    u = np.zeros((Nt, Ny, Nx))
    for i, ti in enumerate(t):
        a = epsilon * np.sin(omega * ti)
        b = 1.0 - 2.0 * epsilon * np.sin(omega * ti)
        f = a * X**2 + b * X
        u[i] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * Y)

    return {
        "q": u.reshape(Nt, -1),
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "metadata": {
            "name": "Double Gyre",
            "period": period,
            "expected_freq": 1.0 / period,
        },
    }


def generate_taylor_green(
    Nx: int = 64,
    Ny: int = 64,
    Nt: int = 100,
    nu: float = 0.01,
    U0: float = 1.0,
    L: float = 2.0 * np.pi,
) -> dict[str, Any]:
    """Generate a Taylor-Green vortex and return the ``u`` snapshots."""
    t_max = 3.0 / (2.0 * nu)

    x = np.linspace(0.0, L, Nx, endpoint=False)
    y = np.linspace(0.0, L, Ny, endpoint=False)
    t = np.linspace(0.0, t_max, Nt)
    dt = float(t[1] - t[0])
    X, Y = np.meshgrid(x, y)  # contract layout: (Ny, Nx), index = iy*Nx + ix

    u = np.zeros((Nt, Ny, Nx))
    decay = np.exp(-2.0 * nu * t)
    for i, d in enumerate(decay):
        u[i] = -U0 * np.cos(X) * np.sin(Y) * d

    return {
        "q": u.reshape(Nt, -1),
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "metadata": {
            "name": "Taylor-Green Vortex",
            "decay_rate": 2.0 * nu,
            "dmd_eigenvalue": float(np.exp(-2.0 * nu * dt)),
        },
    }


def generate_cylinder_wake(
    Nx: int = 100,
    Ny: int = 50,
    Nt: int = 500,
    Re: float = 100.0,
    D: float = 1.0,
    U_inf: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic von Karman cylinder wake and return the ``u`` snapshots."""
    St = 0.212 * (1.0 - 21.2 / Re)
    f_shed = St * U_inf / D
    omega = 2.0 * np.pi * f_shed
    T_shed = 1.0 / f_shed
    t_max = 10.0 * T_shed

    x = np.linspace(0.0, 10.0, Nx)
    y = np.linspace(-2.5, 2.5, Ny)
    t = np.linspace(0.0, t_max, Nt)
    dt = float(t[1] - t[0])
    X, Y = np.meshgrid(x, y)  # contract layout: (Ny, Nx), index = iy*Nx + ix

    x_cyl = 1.0
    wake_width = D * (1.0 + 0.1 * np.sqrt(np.maximum(X - x_cyl, 0.0)))
    wake_decay = np.exp(-0.1 * np.maximum(X - x_cyl, 0.0))
    amp = 0.3 * U_inf * wake_decay

    u = np.zeros((Nt, Ny, Nx))
    rng = np.random.default_rng(seed=seed)
    for i, ti in enumerate(t):
        phase = omega * ti
        k_x = omega / (0.8 * U_inf)
        spatial_phase = k_x * (X - x_cyl)
        y_envelope = np.exp(-(Y**2) / (2.0 * wake_width**2))
        u[i] = U_inf * (1.0 - 0.5 * wake_decay * y_envelope)
        u[i] += amp * np.sin(phase - spatial_phase) * y_envelope * (Y / wake_width)
    u += rng.standard_normal(u.shape) * 0.02 * U_inf

    return {
        "q": u.reshape(Nt, -1),
        "x": x,
        "y": y,
        "z": None,
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": 1,
        "Ns": Nt,
        "seed": seed,
        "metadata": {
            "name": "Cylinder Wake",
            "Re": Re,
            "St": St,
            "f_shed": f_shed,
        },
    }


GENERATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "double_gyre": generate_double_gyre,
    "taylor_green": generate_taylor_green,
    "cylinder_wake": generate_cylinder_wake,
}


def generate_example_dataset(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create one of the built-in synthetic example datasets."""
    if name not in GENERATORS:
        raise ValueError(f"Unknown built-in generator '{name}'. Available: {sorted(GENERATORS)}")
    return GENERATORS[name](**(params or {}))
