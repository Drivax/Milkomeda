"""
integrator.py — Leapfrog (Störmer-Verlet) symplectic integrator.

Update equations (kick-drift-kick form):
  v_{i+1/2} = v_{i-1/2} + a_i * Δt          (kick)
  x_{i+1}   = x_i + v_{i+1/2} * Δt          (drift)
  a_{i+1}   = f(x_{i+1})                     (force update)
  v_{i+1}   = v_{i+1/2} + a_{i+1} * Δt/2    (kick again)

Full KDK (Kick-Drift-Kick) variant is used: it produces synchronised
positions and velocities at every full step, which is required for
energy monitoring and HDF5 snapshots.

Units: kpc, M_sun, yr
"""

from __future__ import annotations
import numpy as np
from octree import compute_accelerations, compute_accelerations_direct


def leapfrog_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    dt: float,
    softening: float = 0.1,
    theta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance the system by one timestep using KDK Leapfrog.

    Parameters
    ----------
    pos       : (N, 3) positions in kpc
    vel       : (N, 3) velocities in kpc/yr
    acc       : (N, 3) accelerations at current positions in kpc/yr²
    mass      : (N,)   masses in M_sun
    dt        : timestep in yr
    softening : softening length in kpc
    theta     : Barnes-Hut opening angle

    Returns
    -------
    new_pos, new_vel, new_acc — all shape matching inputs
    """
    half_dt = 0.5 * dt

    # Half-kick
    vel_half = vel + acc * half_dt

    # Drift
    new_pos = pos + vel_half * dt

    # Recompute forces
    new_acc = compute_accelerations(new_pos, mass, softening=softening, theta=theta)

    # Half-kick again
    new_vel = vel_half + new_acc * half_dt

    return new_pos, new_vel, new_acc


def leapfrog_step_direct(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    dt: float,
    softening: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDK Leapfrog using the fast numpy direct (O(N²)) force solver."""
    half_dt = 0.5 * dt
    vel_half = vel + acc * half_dt
    new_pos = pos + vel_half * dt
    new_acc = compute_accelerations_direct(new_pos, mass, softening=softening)
    new_vel = vel_half + new_acc * half_dt
    return new_pos, new_vel, new_acc


def compute_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    softening: float = 0.1,
) -> tuple[float, float]:
    """Compute total kinetic and potential energy.

    Returns
    -------
    (KE, PE) in M_sun * kpc² / yr²
    """
    from octree import G_CONST

    # Kinetic energy
    v2 = (vel ** 2).sum(axis=1)
    KE = 0.5 * (mass * v2).sum()

    # Potential energy — direct O(N²) only for diagnostics on small N
    N = pos.shape[0]
    PE = 0.0
    for i in range(N):
        dr = pos[i + 1 :] - pos[i]
        dist = np.sqrt((dr ** 2).sum(axis=1) + softening ** 2)
        PE -= G_CONST * mass[i] * (mass[i + 1 :] / dist).sum()

    return float(KE), float(PE)
