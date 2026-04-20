"""
sweep_transverse_velocity.py

Sweep Andromeda transverse velocity and quantify uncertainty bands from
multiple stochastic runs per velocity.

Outputs:
- <output-prefix>_summary.csv : percentile summaries vs v_trans
- <output-prefix>_bands.png   : confidence-band figure
- <output-prefix>_raw.npz     : raw sampled metrics and traces

Example:
    python sweep_transverse_velocity.py \
        --vtrans-min 0 --vtrans-max 80 --vtrans-count 9 \
        --runs-per-v 24 --N 120 --steps 500 --dt 5e6 \
        --sample-every 5 --method auto --output-prefix vtrans_sweep
"""

from __future__ import annotations

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from galaxy import build_initial_conditions
from integrator import leapfrog_step, leapfrog_step_direct
from octree import compute_accelerations, compute_accelerations_direct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep Andromeda transverse velocity and build uncertainty bands"
    )
    p.add_argument("--vtrans-min", type=float, default=0.0,
                   help="Minimum transverse velocity in km/s (default 0)")
    p.add_argument("--vtrans-max", type=float, default=80.0,
                   help="Maximum transverse velocity in km/s (default 80)")
    p.add_argument("--vtrans-count", type=int, default=9,
                   help="Number of velocity points in the sweep (default 9)")
    p.add_argument("--andromeda-radial-kms", type=float, default=-110.0,
                   help="Andromeda radial velocity in km/s (default -110)")
    p.add_argument("--initial-distance-kpc", type=float, default=785.0,
                   help="Initial MW-M31 center separation in kpc (default 785.0)")
    p.add_argument("--runs-per-v", type=int, default=16,
                   help="Monte Carlo runs per transverse velocity (default 16)")
    p.add_argument("--N", type=int, default=120,
                   help="Particles per galaxy per run (default 120)")
    p.add_argument("--dt", type=float, default=5e6,
                   help="Timestep in years (default 5e6)")
    p.add_argument("--steps", type=int, default=500,
                   help="Integration steps per run (default 500)")
    p.add_argument("--sample-every", type=int, default=5,
                   help="Sample COM separation every N steps (default 5)")
    p.add_argument("--softening", type=float, default=1.0,
                   help="Softening length in kpc (default 1.0)")
    p.add_argument("--theta", type=float, default=0.5,
                   help="Barnes-Hut opening angle (default 0.5)")
    p.add_argument("--method", type=str, default="auto", choices=["auto", "direct", "bh"],
                   help="Force method (default auto)")
    p.add_argument("--close-threshold", type=float, default=100.0,
                   help="Close-approach threshold in kpc (default 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default 42)")
    p.add_argument("--output-prefix", type=str, default="vtrans_sweep",
                   help="Prefix for output files (default vtrans_sweep)")
    p.add_argument("--dpi", type=int, default=180,
                   help="Figure dpi (default 180)")
    return p.parse_args()


def _mass_weighted_com(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return (mass[:, None] * pos).sum(axis=0) / (mass.sum() + 1e-300)


def run_one_case(
    N: int,
    dt: float,
    steps: int,
    sample_every: int,
    softening: float,
    theta: float,
    method: str,
    seed: int,
    andromeda_radial_kms: float,
    andromeda_transverse_kms: float,
    initial_distance_kpc: float,
    close_threshold_kpc: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Run one simulation and return COM separation trace + summary metrics."""
    pos, vel, mass = build_initial_conditions(
        N,
        seed=seed,
        andromeda_radial_kms=andromeda_radial_kms,
        andromeda_transverse_kms=andromeda_transverse_kms,
        initial_distance_kpc=initial_distance_kpc,
    )
    n_total = pos.shape[0]
    n_half = n_total // 2
    mass_mw = mass[:n_half]
    mass_m31 = mass[n_half:]

    use_direct = (method == "direct") or (method == "auto" and n_total <= 3000)
    if use_direct:
        acc = compute_accelerations_direct(pos, mass, softening=softening)
    else:
        acc = compute_accelerations(pos, mass, softening=softening, theta=theta)

    n_samples = steps // sample_every + 1
    t_gyr = np.empty(n_samples, dtype=np.float64)
    sep_kpc = np.empty(n_samples, dtype=np.float64)

    # t = 0
    i_sample = 0
    com_mw = _mass_weighted_com(pos[:n_half], mass_mw)
    com_m31 = _mass_weighted_com(pos[n_half:], mass_m31)
    sep0 = np.linalg.norm(com_m31 - com_mw)
    t_gyr[i_sample] = 0.0
    sep_kpc[i_sample] = sep0
    min_sep = float(sep0)
    t_min = 0.0
    first_close = -1.0 if sep0 > close_threshold_kpc else 0.0
    i_sample += 1

    t_sim = 0.0
    for step in range(1, steps + 1):
        if use_direct:
            pos, vel, acc = leapfrog_step_direct(
                pos, vel, acc, mass, dt=dt, softening=softening
            )
        else:
            pos, vel, acc = leapfrog_step(
                pos, vel, acc, mass, dt=dt, softening=softening, theta=theta
            )

        t_sim += dt
        if step % sample_every == 0:
            t_now = t_sim / 1e9
            com_mw = _mass_weighted_com(pos[:n_half], mass_mw)
            com_m31 = _mass_weighted_com(pos[n_half:], mass_m31)
            sep = float(np.linalg.norm(com_m31 - com_mw))
            t_gyr[i_sample] = t_now
            sep_kpc[i_sample] = sep
            if sep < min_sep:
                min_sep = sep
                t_min = t_now
            if first_close < 0.0 and sep <= close_threshold_kpc:
                first_close = t_now
            i_sample += 1

    return t_gyr, sep_kpc, min_sep, t_min, first_close


def main() -> None:
    args = parse_args()

    v_values = np.linspace(args.vtrans_min, args.vtrans_max, args.vtrans_count)
    n_samples = args.steps // args.sample_every + 1

    # Raw arrays for later analysis.
    all_sep = np.empty((args.vtrans_count, args.runs_per_v, n_samples), dtype=np.float64)
    all_min_sep = np.empty((args.vtrans_count, args.runs_per_v), dtype=np.float64)
    all_t_min = np.empty((args.vtrans_count, args.runs_per_v), dtype=np.float64)
    all_t_close = np.empty((args.vtrans_count, args.runs_per_v), dtype=np.float64)

    print("=" * 72)
    print("Milkomeda Transverse-Velocity Uncertainty Sweep")
    print("=" * 72)
    print(f"v_trans range  : {args.vtrans_min:.1f} .. {args.vtrans_max:.1f} km/s")
    print(f"v points       : {args.vtrans_count}")
    print(f"runs per point : {args.runs_per_v}")
    print(f"N/galaxy       : {args.N}")
    print(f"steps          : {args.steps}")
    print(f"dt (yr)        : {args.dt:.2e}")
    print(f"duration (Gyr) : {args.steps * args.dt / 1e9:.2f}")
    print(f"method         : {args.method}")
    print(f"M31 v_rad      : {args.andromeda_radial_kms:.2f} km/s")
    print(f"init dist      : {args.initial_distance_kpc:.2f} kpc")
    print(f"close threshold: {args.close_threshold:.1f} kpc")
    print("=" * 72)

    t0 = time.perf_counter()
    t_ref = None
    total_runs = args.vtrans_count * args.runs_per_v
    with tqdm(total=total_runs, desc="Sweep runs", ncols=90) as pbar:
        for i_v, v_trans in enumerate(v_values):
            for i_run in range(args.runs_per_v):
                seed = args.seed + i_v * 100000 + i_run
                t_gyr, sep_kpc, min_sep, t_min, t_close = run_one_case(
                    N=args.N,
                    dt=args.dt,
                    steps=args.steps,
                    sample_every=args.sample_every,
                    softening=args.softening,
                    theta=args.theta,
                    method=args.method,
                    seed=seed,
                    andromeda_radial_kms=args.andromeda_radial_kms,
                    andromeda_transverse_kms=float(v_trans),
                    initial_distance_kpc=args.initial_distance_kpc,
                    close_threshold_kpc=args.close_threshold,
                )
                if t_ref is None:
                    t_ref = t_gyr
                all_sep[i_v, i_run, :] = sep_kpc
                all_min_sep[i_v, i_run] = min_sep
                all_t_min[i_v, i_run] = t_min
                all_t_close[i_v, i_run] = t_close
                pbar.update(1)

    # Percentile summaries per transverse-velocity point.
    min_p16, min_p50, min_p84 = np.percentile(all_min_sep, [16, 50, 84], axis=1)
    tmin_p16, tmin_p50, tmin_p84 = np.percentile(all_t_min, [16, 50, 84], axis=1)

    # Close-approach probability within simulated time.
    close_mask = all_t_close >= 0.0
    close_prob = close_mask.mean(axis=1)

    summary_csv = f"{args.output_prefix}_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "v_transverse_kms",
            "min_sep_p16_kpc", "min_sep_p50_kpc", "min_sep_p84_kpc",
            "t_min_sep_p16_gyr", "t_min_sep_p50_gyr", "t_min_sep_p84_gyr",
            "p_close_approach",
        ])
        for i, v in enumerate(v_values):
            writer.writerow([
                float(v),
                float(min_p16[i]), float(min_p50[i]), float(min_p84[i]),
                float(tmin_p16[i]), float(tmin_p50[i]), float(tmin_p84[i]),
                float(close_prob[i]),
            ])

    # Save raw arrays for downstream analysis.
    raw_npz = f"{args.output_prefix}_raw.npz"
    np.savez_compressed(
        raw_npz,
        v_transverse_kms=v_values,
        time_gyr=t_ref,
        separation_kpc=all_sep,
        min_separation_kpc=all_min_sep,
        t_min_separation_gyr=all_t_min,
        t_close_approach_gyr=all_t_close,
        close_threshold_kpc=args.close_threshold,
    )

    # Figure: confidence bands versus v_trans.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), facecolor="white")

    ax1.fill_between(v_values, min_p16, min_p84, color="#9ed0ff", alpha=0.55, linewidth=0)
    ax1.plot(v_values, min_p50, color="#005a9c", lw=2.2, label="Median min separation")
    ax1.scatter(v_values, min_p50, color="#005a9c", s=24)
    ax1.set_ylabel("Minimum COM separation (kpc)")
    ax1.set_title("Milkomeda uncertainty bands vs Andromeda transverse velocity")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(frameon=False)

    ax2.fill_between(v_values, tmin_p16, tmin_p84, color="#f8c9a4", alpha=0.55, linewidth=0)
    ax2.plot(v_values, tmin_p50, color="#c75000", lw=2.2, label="Median time of min separation")
    ax2.scatter(v_values, tmin_p50, color="#c75000", s=24)
    ax2.set_xlabel("Andromeda transverse velocity (km/s)")
    ax2.set_ylabel("Time of min separation (Gyr)")
    ax2.grid(alpha=0.25, linestyle="--")

    ax2b = ax2.twinx()
    ax2b.plot(v_values, close_prob, color="#2e7d32", lw=1.8, linestyle=":", label="Close-approach probability")
    ax2b.set_ylabel("P(close approach)")
    ax2b.set_ylim(-0.02, 1.02)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig_path = f"{args.output_prefix}_bands.png"
    fig.savefig(fig_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    print(f"Saved: {summary_csv}")
    print(f"Saved: {raw_npz}")
    print(f"Saved: {fig_path}")
    print(f"Completed in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
