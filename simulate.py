"""
simulate.py — Main simulation driver for the Milkomeda galaxy collision.

Usage:
    python simulate.py --N 5000 --dt 1e6 --steps 2000 --theta 0.5

All physical quantities use:
    Length  : kpc
    Mass    : M_sun
    Time    : yr
    Velocity: kpc/yr
"""

import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from galaxy import build_initial_conditions, get_scenario_names, resolve_andromeda_velocities
from integrator import leapfrog_step, leapfrog_step_direct, compute_energy
from octree import compute_accelerations, compute_accelerations_direct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milkomeda — Milky Way × Andromeda N-body collision simulator"
    )
    p.add_argument("--N", type=int, default=5000,
                   help="Number of particles per galaxy (default 5000)")
    p.add_argument("--dt", type=float, default=1e6,
                   help="Timestep in years (default 1e6)")
    p.add_argument("--steps", type=int, default=2000,
                   help="Number of integration steps (default 2000)")
    p.add_argument("--theta", type=float, default=0.5,
                   help="Barnes-Hut opening angle (default 0.5)")
    p.add_argument("--softening", type=float, default=0.1,
                   help="Softening length in kpc (default 0.1)")
    p.add_argument("--output", type=str, default="output.h5",
                   help="Output HDF5 file (default output.h5)")
    p.add_argument("--snapshot-every", type=int, default=10,
                   help="Save snapshot every N steps (default 10)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for initial conditions (default 42)")
    p.add_argument("--scenario", type=str, default="baseline",
                   choices=get_scenario_names(),
                   help="Named initial-condition preset (default baseline)")
    p.add_argument("--andromeda-radial-kms", type=float, default=None,
                   help="Override scenario radial velocity in km/s")
    p.add_argument("--andromeda-transverse-kms", type=float, default=None,
                   help="Override scenario transverse velocity in km/s")
    p.add_argument("--initial-distance-kpc", type=float, default=785.0,
                   help="Initial MW-M31 center separation in kpc (default 785.0)")
    p.add_argument("--validate", action="store_true",
                   help="Compute energy at each snapshot for validation")
    p.add_argument("--method", type=str, default="auto",
                   choices=["auto", "direct", "bh"],
                   help="Force method: 'direct' (numpy O(N^2)), 'bh' (Barnes-Hut), "
                        "'auto' = direct when N_total<=3000 else bh (default auto)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def create_output_file(path: str, N_total: int, n_snapshots: int,
                       args: argparse.Namespace,
                       scenario_name: str,
                       andromeda_radial_kms: float,
                       andromeda_transverse_kms: float,
                       initial_distance_kpc: float) -> h5py.File:
    f = h5py.File(path, "w")
    meta = f.create_group("metadata")
    meta.attrs["N_per_galaxy"] = args.N
    meta.attrs["N_total"] = N_total
    meta.attrs["dt_yr"] = args.dt
    meta.attrs["steps"] = args.steps
    meta.attrs["theta"] = args.theta
    meta.attrs["softening_kpc"] = args.softening
    meta.attrs["snapshot_every"] = args.snapshot_every
    meta.attrs["scenario"] = scenario_name
    meta.attrs["andromeda_radial_kms"] = andromeda_radial_kms
    meta.attrs["andromeda_transverse_kms"] = andromeda_transverse_kms
    meta.attrs["initial_distance_kpc"] = initial_distance_kpc

    f.create_dataset("pos",  shape=(n_snapshots, N_total, 3), dtype="float32")
    f.create_dataset("vel",  shape=(n_snapshots, N_total, 3), dtype="float32")
    f.create_dataset("mass", shape=(N_total,),               dtype="float64")
    f.create_dataset("time", shape=(n_snapshots,),            dtype="float64")

    if args.validate:
        f.create_dataset("KE",    shape=(n_snapshots,), dtype="float64")
        f.create_dataset("PE",    shape=(n_snapshots,), dtype="float64")
        f.create_dataset("E_tot", shape=(n_snapshots,), dtype="float64")

    return f


def _mass_weighted_com(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return (mass[:, None] * pos).sum(axis=0) / (mass.sum() + 1e-300)


def write_snapshot(f: h5py.File, snap_idx: int, pos: np.ndarray,
                   vel: np.ndarray, t: float,
                   KE: float = 0.0, PE: float = 0.0,
                   validate: bool = False) -> None:
    f["pos"][snap_idx] = pos.astype("float32")
    f["vel"][snap_idx] = vel.astype("float32")
    f["time"][snap_idx] = t
    if validate:
        f["KE"][snap_idx] = KE
        f["PE"][snap_idx] = PE
        f["E_tot"][snap_idx] = KE + PE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scenario_name, radial_kms, transverse_kms = resolve_andromeda_velocities(
        args.scenario,
        andromeda_radial_kms=args.andromeda_radial_kms,
        andromeda_transverse_kms=args.andromeda_transverse_kms,
    )

    print("=" * 60)
    print("  Milkomeda — Galaxy Collision Simulator")
    print("=" * 60)
    print(f"  N per galaxy  : {args.N:,}")
    print(f"  N total       : {2 * args.N:,}")
    print(f"  dt            : {args.dt:.2e} yr")
    print(f"  Steps         : {args.steps:,}")
    print(f"  Sim duration  : {args.dt * args.steps / 1e9:.2f} Gyr")
    print(f"  theta         : {args.theta}")
    print(f"  softening     : {args.softening} kpc")
    print(f"  Method        : {args.method}")
    print(f"  Scenario      : {scenario_name}")
    print(f"  M31 v_rad     : {radial_kms:.2f} km/s")
    print(f"  M31 v_trans   : {transverse_kms:.2f} km/s")
    print(f"  Initial dist  : {args.initial_distance_kpc:.2f} kpc")
    print(f"  Output        : {args.output}")
    print("=" * 60)

    # --- Build initial conditions ---
    print("\n[1/4] Building initial conditions...")
    t0 = time.perf_counter()
    pos, vel, mass = build_initial_conditions(
        args.N,
        seed=args.seed,
        scenario=scenario_name,
        andromeda_radial_kms=radial_kms,
        andromeda_transverse_kms=transverse_kms,
        initial_distance_kpc=args.initial_distance_kpc,
    )
    N_total = pos.shape[0]
    N_half = N_total // 2
    mass_mw = mass[:N_half]
    mass_and = mass[N_half:]
    print(f"      {N_total:,} particles initialised in {time.perf_counter()-t0:.1f}s")

    # --- Decide force method ---
    use_direct = (
        args.method == "direct"
        or (args.method == "auto" and N_total <= 3000)
    )
    method_label = "direct (numpy O(N^2))" if use_direct else f"Barnes-Hut (theta={args.theta})"

    # --- Compute initial forces ---
    print(f"[2/4] Computing initial forces ({method_label})...")
    t0 = time.perf_counter()
    if use_direct:
        acc = compute_accelerations_direct(pos, mass, softening=args.softening)
    else:
        acc = compute_accelerations(pos, mass, softening=args.softening, theta=args.theta)
    print(f"      Done in {time.perf_counter()-t0:.1f}s")

    # --- Set up output ---
    n_snaps = args.steps // args.snapshot_every + 1
    print(f"[3/4] Opening output file '{args.output}' ({n_snaps} snapshots)...")
    hf = create_output_file(
        args.output,
        N_total,
        n_snaps,
        args,
        scenario_name=scenario_name,
        andromeda_radial_kms=radial_kms,
        andromeda_transverse_kms=transverse_kms,
        initial_distance_kpc=args.initial_distance_kpc,
    )
    hf["mass"][:] = mass

    # Write t=0 snapshot
    snap_idx = 0
    KE0, PE0 = 0.0, 0.0
    if args.validate:
        KE0, PE0 = compute_energy(pos, vel, mass, softening=args.softening)
        E0 = KE0 + PE0
        print(f"      E(t=0): KE={KE0:.4e}  PE={PE0:.4e}  E={E0:.4e}")
    write_snapshot(hf, snap_idx, pos, vel, 0.0, KE0, PE0, args.validate)
    snap_idx += 1

    # --- Integration loop ---
    print("[4/4] Running simulation...")
    t_sim = 0.0
    E0_total = KE0 + PE0
    # Track summary metrics for post-run analysis.
    com_mw = _mass_weighted_com(pos[:N_half], mass_mw)
    com_and = _mass_weighted_com(pos[N_half:], mass_and)
    separation = np.linalg.norm(com_and - com_mw)
    min_separation_kpc = float(separation)
    min_separation_time_gyr = 0.0
    close_approach_threshold_kpc = 100.0
    first_close_approach_time_gyr = -1.0
    max_energy_drift_pct = 0.0

    with tqdm(total=args.steps, unit="step", ncols=72) as pbar:
        for step in range(1, args.steps + 1):
            if use_direct:
                pos, vel, acc = leapfrog_step_direct(
                    pos, vel, acc, mass,
                    dt=args.dt,
                    softening=args.softening,
                )
            else:
                pos, vel, acc = leapfrog_step(
                    pos, vel, acc, mass,
                    dt=args.dt,
                    softening=args.softening,
                    theta=args.theta,
                )
            t_sim += args.dt

            if step % args.snapshot_every == 0:
                KE, PE = 0.0, 0.0
                if args.validate:
                    KE, PE = compute_energy(pos, vel, mass, softening=args.softening)
                    E_now = KE + PE
                    drift_pct = abs((E_now - E0_total) / (E0_total + 1e-300)) * 100
                    max_energy_drift_pct = max(max_energy_drift_pct, float(drift_pct))
                    pbar.set_postfix({"E_drift%": f"{drift_pct:.3f}"})

                com_mw = _mass_weighted_com(pos[:N_half], mass_mw)
                com_and = _mass_weighted_com(pos[N_half:], mass_and)
                separation = float(np.linalg.norm(com_and - com_mw))
                t_gyr = t_sim / 1e9
                if separation < min_separation_kpc:
                    min_separation_kpc = separation
                    min_separation_time_gyr = t_gyr
                if first_close_approach_time_gyr < 0.0 and separation <= close_approach_threshold_kpc:
                    first_close_approach_time_gyr = t_gyr

                write_snapshot(hf, snap_idx, pos, vel, t_sim, KE, PE, args.validate)
                snap_idx += 1

            pbar.update(1)

    meta = hf["metadata"]
    meta.attrs["min_com_separation_kpc"] = min_separation_kpc
    meta.attrs["min_com_separation_time_gyr"] = min_separation_time_gyr
    meta.attrs["close_approach_threshold_kpc"] = close_approach_threshold_kpc
    meta.attrs["first_close_approach_time_gyr"] = first_close_approach_time_gyr
    if args.validate:
        meta.attrs["max_energy_drift_pct"] = max_energy_drift_pct

    hf.close()
    print(f"\nDone. Simulation saved to '{args.output}'.")
    print(f"Simulated {t_sim/1e9:.3f} Gyr of cosmic time.")
    print(f"Minimum COM separation: {min_separation_kpc:.2f} kpc at t={min_separation_time_gyr:.3f} Gyr")
    if first_close_approach_time_gyr >= 0.0:
        print(
            f"First COM close approach (<={close_approach_threshold_kpc:.0f} kpc): "
            f"t={first_close_approach_time_gyr:.3f} Gyr"
        )
    else:
        print(
            f"No COM close approach <= {close_approach_threshold_kpc:.0f} kpc "
            "within simulated time span"
        )
    if args.validate:
        print(f"Max energy drift across snapshots: {max_energy_drift_pct:.4f}%")


if __name__ == "__main__":
    main()
