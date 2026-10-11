"""
galaxy.py — Initial conditions generator for Milky Way and Andromeda.

Generates particle distributions following:
  - NFW dark matter halo
  - Hernquist bulge profile
  - approximate disk-like profile

All units are:
  - Length: kpc
  - Mass:   solar masses (M_sun)
  - Time:   years (yr)
  - Velocity: kpc / yr

Physical constants:
    G = 4.498 × 10⁻²⁴ kpc³ / (M_sun * yr²)
"""

import numpy as np
from dataclasses import dataclass

from src.config import G_CONST, KMS_TO_KPCYR


@dataclass
class GalaxyParams:
    """Physical parameters for a galaxy."""
    # Total masses in M_sun
    M_halo: float        # NFW dark matter halo mass
    M_bulge: float       # Hernquist bulge mass
    M_disk: float        # Miyamoto-Nagai disk mass
    # Scale radii in kpc
    r_s: float           # NFW scale radius
    a_bulge: float       # Hernquist scale length
    a_disk: float        # Miyamoto-Nagai disk scale length
    b_disk: float        # Miyamoto-Nagai disk scale height
    # Truncation radius in kpc
    r_trunc: float       # outer cutoff for NFW sampling


def _sample_hernquist(N, M, a, rng):
    """Sample positions from a Hernquist profile using inverse CDF.

    ρ(r) = M/(2π) * a / [r * (r+a)³]
    CDF:  M(<r) = M * r² / (r+a)²
    """
    u = rng.uniform(0.0, 1.0, N)
    r = a * np.sqrt(u) / (1.0 - np.sqrt(u))
    # Clip unphysical infinities from u very close to 1
    r = np.clip(r, 0.0, 1e4)
    phi = rng.uniform(0, 2 * np.pi, N)
    cos_theta = rng.uniform(-1.0, 1.0, N)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    pos = np.column_stack([
        r * sin_theta * np.cos(phi),
        r * sin_theta * np.sin(phi),
        r * cos_theta,
    ])
    return pos


def _sample_nfw_radius(N, r_s, r_trunc, rng):
    """Invert enclosed NFW mass; radial density includes the spherical volume factor."""
    x=np.linspace(0,r_trunc/r_s,20000)
    cdf=np.log1p(x)-x/(1+x)
    return r_s*np.interp(rng.random(N),cdf/cdf[-1],x)


def _sample_disk_miyamoto_nagai(N, M, a, b, rng):
    """Sample positions in a approximate disk-like profile."""
    # Sample cylindrical R from surface density Σ(R) ∝ R * (a + b) / (R² + (a+b)²)^(3/2)
    # Use inverse-CDF: CDF ∝ 1 - (a+b)/sqrt(R²+(a+b)²)
    ab = a + b
    u = rng.uniform(0.0, 1.0, N)
    R = ab * np.sqrt(1.0 / (1.0 - u) ** 2 - 1.0)
    R = np.clip(R, 0.0, 1e3)
    phi = rng.uniform(0, 2 * np.pi, N)
    # z from exponential-like distribution with scale b
    z = rng.laplace(0.0, b / 2.0, N)
    pos = np.column_stack([
        R * np.cos(phi),
        R * np.sin(phi),
        z,
    ])
    return pos


def _circular_velocity_hernquist(r, M, a):
    """Circular velocity from Hernquist potential: v_c² = GM*r/(r+a)²."""
    return np.sqrt(G_CONST * M * r / (r + a) ** 2)


def _circular_velocity_nfw(r, M_halo, r_s, r_trunc):
    """NFW circular velocity contribution (approximate)."""
    x = r / r_s
    # NFW enclosed mass (unnormalized): ln(1+x) - x/(1+x)
    # Normalize so M(<r_trunc) = M_halo
    x_trunc = r_trunc / r_s
    norm = np.log(1.0 + x_trunc) - x_trunc / (1.0 + x_trunc)
    M_enc = M_halo * (np.log(1.0 + x) - x / (1.0 + x)) / (norm + 1e-30)
    M_enc = np.clip(M_enc, 0.0, None)
    return np.sqrt(G_CONST * M_enc / (r + 1e-10))


def _assign_velocities(pos, params: GalaxyParams, rng, dispersion):
    """Assign approximate rotating velocities; not a self-consistent equilibrium."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(R ** 2 + z ** 2) + 1e-10

    v_halo = _circular_velocity_nfw(r, params.M_halo, params.r_s, params.r_trunc)
    v_bulge = _circular_velocity_hernquist(r, params.M_bulge, params.a_bulge)
    v_disk = _circular_velocity_hernquist(r, params.M_disk, params.a_disk)

    v_c = np.sqrt(v_halo ** 2 + v_bulge ** 2 + v_disk ** 2)

    # Tangential velocity in x-y plane
    phi = np.arctan2(y, x)
    # Add small dispersion
    sigma = dispersion * v_c
    vx = -v_c * np.sin(phi) + rng.normal(0.0, sigma)
    vy = v_c * np.cos(phi) + rng.normal(0.0, sigma)
    vz = rng.normal(0.0, sigma * 0.5)
    return np.column_stack([vx, vy, vz])


def _rotation_matrix_x(angle_deg):
    """Rotation matrix around X axis."""
    a = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)],
    ])


def build_galaxy(params: GalaxyParams, N_total: int, center: np.ndarray,
                 velocity: np.ndarray, inclination_deg: float = 0.0,
                 seed: int = 42, dispersion: float = 0.15) -> dict:
    """Build a galaxy particle distribution.

    Parameters
    ----------
    params       : GalaxyParams for this galaxy
    N_total      : total number of particles (split across halo/bulge/disk)
    center       : (3,) array, position of galaxy center in kpc
    velocity     : (3,) array, bulk velocity in kpc/yr
    inclination_deg : disk inclination angle in degrees
    seed         : RNG seed

    Returns
    -------
    dict with keys 'pos', 'vel', 'mass' — each shape (N, 3) or (N,)
    """
    rng = np.random.default_rng(seed)

    # Partition particles: 60% halo, 15% bulge, 25% disk
    N_halo = int(0.60 * N_total)
    N_bulge = int(0.15 * N_total)
    N_disk = N_total - N_halo - N_bulge

    # --- Sample positions ---
    # Halo
    r_halo = _sample_nfw_radius(N_halo, params.r_s, params.r_trunc, rng)
    phi_h = rng.uniform(0, 2 * np.pi, N_halo)
    cos_t = rng.uniform(-1.0, 1.0, N_halo)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    pos_halo = np.column_stack([
        r_halo * sin_t * np.cos(phi_h),
        r_halo * sin_t * np.sin(phi_h),
        r_halo * cos_t,
    ])

    # Bulge (Hernquist)
    pos_bulge = _sample_hernquist(N_bulge, params.M_bulge, params.a_bulge, rng)

    # Disk (Miyamoto-Nagai)
    pos_disk = _sample_disk_miyamoto_nagai(N_disk, params.M_disk, params.a_disk, params.b_disk, rng)

    pos = np.vstack([pos_halo, pos_bulge, pos_disk])

    # --- Masses ---
    m_halo_part = params.M_halo / N_halo
    m_bulge_part = params.M_bulge / N_bulge
    m_disk_part = params.M_disk / N_disk
    mass = np.concatenate([
        np.full(N_halo, m_halo_part),
        np.full(N_bulge, m_bulge_part),
        np.full(N_disk, m_disk_part),
    ])

    # --- Velocities ---
    vel_halo = _assign_velocities(pos_halo, params, rng, dispersion)
    vel_bulge = _assign_velocities(pos_bulge, params, rng, dispersion)
    vel_disk = _assign_velocities(pos_disk, params, rng, dispersion)
    vel = np.vstack([vel_halo, vel_bulge, vel_disk])

    # --- Apply inclination rotation ---
    R = _rotation_matrix_x(inclination_deg)
    pos = pos @ R.T
    vel = vel @ R.T

    # --- Translate to center / add bulk velocity ---
    pos -= np.average(pos,axis=0,weights=mass)
    vel -= np.average(vel,axis=0,weights=mass)
    pos += center
    vel += velocity

    return {"pos": pos, "vel": vel, "mass": mass}


def build_initial_conditions(config):
    """Sample both finite particle systems, enforce requested relative COM state."""
    # Component fractions and relative structural scales retain the original model.
    # These approximate rotating ICs are not an equilibrium distribution function.
    def structure(total, radius, which):
        fractions=(100,1,5) if which=="mw" else (150,2,8)
        masses=[total*f/sum(fractions) for f in fractions]
        ratios=(.035,.175,.0175,10) if which=="mw" else (.04,.2,.02,10)
        return GalaxyParams(*masses,radius,*(radius*q for q in ratios))
    mw=build_galaxy(structure(config.mw_mass,config.mw_radius_kpc,"mw"),config.N,
                    np.zeros(3),np.zeros(3),config.mw_inclination_deg,config.seed,config.dispersion)
    m31=build_galaxy(structure(config.m31_mass,config.m31_radius_kpc,"m31"),config.N,
                     np.array([config.initial_distance_kpc,0,0]),
                     np.array([config.andromeda_radial_kms,config.andromeda_transverse_kms,0])*KMS_TO_KPCYR,
                     config.m31_inclination_deg,config.seed+1,config.dispersion)
    pos=np.vstack([mw["pos"],m31["pos"]]); vel=np.vstack([mw["vel"],m31["vel"]])
    mass=np.concatenate([mw["mass"],m31["mass"]])
    pos-=np.average(pos,axis=0,weights=mass)
    vel-=np.average(vel,axis=0,weights=mass)
    return pos,vel,mass
