"""
galaxy.py — Initial conditions generator for Milky Way and Andromeda.

Generates particle distributions following:
  - NFW dark matter halo
  - Hernquist bulge profile
  - Miyamoto-Nagai disk profile

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
from typing import Optional

# Gravitational constant in kpc^3 M_sun^-1 yr^-2
# Derived from G = 4.3009e-6 kpc (km/s)^2 M_sun^-1 and
# (1 km/s)^2 = (1.022712e-9 kpc/yr)^2.
G_CONST = 4.498e-24

# Unit conversion: km/s -> kpc/yr
KMS_TO_KPCYR = 1.022712e-9  # 1 km/s = 1.022712e-9 kpc/yr


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


# --- Milky Way parameters (from observational data) ---
MILKY_WAY = GalaxyParams(
    M_halo=1.0e12,
    M_bulge=1.0e10,
    M_disk=5.0e10,
    r_s=20.0,
    a_bulge=0.7,
    a_disk=3.5,
    b_disk=0.35,
    r_trunc=200.0,
)

# --- Andromeda (M31) parameters ---
ANDROMEDA = GalaxyParams(
    M_halo=1.5e12,
    M_bulge=2.0e10,
    M_disk=8.0e10,
    r_s=25.0,
    a_bulge=1.0,
    a_disk=5.0,
    b_disk=0.5,
    r_trunc=250.0,
)


# Named Andromeda velocity presets (km/s) for quick scenario demos.
SCENARIO_PRESETS = {
    "baseline": {
        "andromeda_radial_kms": -110.0,
        "andromeda_transverse_kms": 17.0,
    },
    "fast-m31": {
        "andromeda_radial_kms": -140.0,
        "andromeda_transverse_kms": 35.0,
    },
    "low-transverse": {
        "andromeda_radial_kms": -110.0,
        "andromeda_transverse_kms": 8.0,
    },
    "head-on": {
        "andromeda_radial_kms": -110.0,
        "andromeda_transverse_kms": 0.0,
    },
}


def get_scenario_names() -> list[str]:
    """Return supported scenario preset names."""
    return list(SCENARIO_PRESETS.keys())


def resolve_andromeda_velocities(
    scenario: str,
    andromeda_radial_kms: Optional[float] = None,
    andromeda_transverse_kms: Optional[float] = None,
) -> tuple[str, float, float]:
    """Resolve final Andromeda radial/transverse velocities from a preset + overrides."""
    scenario_key = scenario.strip().lower()
    if scenario_key not in SCENARIO_PRESETS:
        options = ", ".join(get_scenario_names())
        raise ValueError(f"Unknown scenario '{scenario}'. Choose one of: {options}")

    preset = SCENARIO_PRESETS[scenario_key]
    radial = preset["andromeda_radial_kms"]
    transverse = preset["andromeda_transverse_kms"]

    # Manual CLI values override preset defaults when provided.
    if andromeda_radial_kms is not None:
        radial = float(andromeda_radial_kms)
    if andromeda_transverse_kms is not None:
        transverse = float(andromeda_transverse_kms)

    return scenario_key, radial, transverse


# ---------------------------------------------------------------------------
# Profile sampling helpers
# ---------------------------------------------------------------------------

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
    """Sample radii from an NFW profile via rejection sampling."""
    r_out = []
    batch = max(N * 4, 10000)
    while len(r_out) < N:
        r_try = rng.uniform(0.0, r_trunc, batch)
        # NFW density (unnormalized): 1 / [x*(1+x)^2], x = r/r_s
        x = r_try / r_s
        rho = 1.0 / (x * (1.0 + x) ** 2 + 1e-10)
        rho_max = 1.0 / (0.01 * (1.01) ** 2)  # approx max near r_s*0.01
        u = rng.uniform(0.0, rho_max, batch)
        accepted = r_try[u < rho]
        r_out.extend(accepted.tolist())
    return np.array(r_out[:N])


def _sample_disk_miyamoto_nagai(N, M, a, b, rng):
    """Sample positions in a Miyamoto-Nagai disk profile."""
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


def _assign_velocities(pos, params: GalaxyParams, rng):
    """Assign approximate equilibrium circular velocities in the disk plane."""
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
    sigma = 0.15 * v_c
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
                 seed: int = 42) -> dict:
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
    vel_halo = _assign_velocities(pos_halo, params, rng)
    vel_bulge = _assign_velocities(pos_bulge, params, rng)
    vel_disk = _assign_velocities(pos_disk, params, rng)
    vel = np.vstack([vel_halo, vel_bulge, vel_disk])

    # --- Apply inclination rotation ---
    R = _rotation_matrix_x(inclination_deg)
    pos = pos @ R.T
    vel = vel @ R.T

    # --- Translate to center / add bulk velocity ---
    pos += center
    vel += velocity

    return {"pos": pos, "vel": vel, "mass": mass}


def build_initial_conditions(
    N: int,
    seed: int = 0,
    scenario: str = "baseline",
    andromeda_radial_kms: Optional[float] = None,
    andromeda_transverse_kms: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build combined initial conditions for MW + Andromeda collision.

    Returns
    -------
    pos  : (2N, 3) float64, positions in kpc
    vel  : (2N, 3) float64, velocities in kpc/yr
    mass : (2N,)   float64, particle masses in M_sun
    """
    # Andromeda is ~785 kpc away. Place MW at origin, Andromeda along +x.
    # Velocities come from a named scenario preset with optional overrides.
    _, radial_kms, transverse_kms = resolve_andromeda_velocities(
        scenario,
        andromeda_radial_kms=andromeda_radial_kms,
        andromeda_transverse_kms=andromeda_transverse_kms,
    )

    mw_center = np.array([0.0, 0.0, 0.0])
    mw_vel = np.array([0.0, 0.0, 0.0])  # MW at rest in COM frame

    and_center = np.array([785.0, 0.0, 0.0])
    # Convert km/s → kpc/yr
    v_rad = radial_kms * KMS_TO_KPCYR
    v_trans = transverse_kms * KMS_TO_KPCYR
    and_vel = np.array([v_rad, v_trans, 0.0])

    # MW disk inclination ~77°
    mw_data = build_galaxy(MILKY_WAY, N, mw_center, mw_vel,
                           inclination_deg=77.0, seed=seed)
    and_data = build_galaxy(ANDROMEDA, N, and_center, and_vel,
                            inclination_deg=0.0, seed=seed + 1)

    pos = np.vstack([mw_data["pos"], and_data["pos"]])
    vel = np.vstack([mw_data["vel"], and_data["vel"]])
    mass = np.concatenate([mw_data["mass"], and_data["mass"]])

    # Shift to center of mass frame
    M_total = mass.sum()
    com_pos = (mass[:, None] * pos).sum(axis=0) / M_total
    com_vel = (mass[:, None] * vel).sum(axis=0) / M_total
    pos -= com_pos
    vel -= com_vel

    return pos, vel, mass
