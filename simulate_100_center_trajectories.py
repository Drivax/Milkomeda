"""
simulate_100_center_trajectories.py

Run the Milky Way-Andromeda simulation many times (default: 100 runs) and
plot all center-of-mass trajectories on a single figure.

Each run uses a different random seed, so you can visualize trajectory spread
from stochastic initial particle sampling.

Example:
    python simulate_100_center_trajectories.py --runs 100 --N 300 --steps 1200 \
        --dt 5e6 --sample-every 10 --method direct --output trajectories_100.png
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from galaxy import build_initial_conditions
from integrator import leapfrog_step, leapfrog_step_direct
from octree import compute_accelerations, compute_accelerations_direct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run many Milkomeda simulations and plot all galaxy-center trajectories "
            "on one figure"
        )
    )
    p.add_argument("--runs", type=int, default=100,
                   help="Number of simulation runs (default 100)")
    p.add_argument("--N", type=int, default=300,
                   help="Particles per galaxy per run (default 300)")
    p.add_argument("--dt", type=float, default=5e6,
                   help="Timestep in years (default 5e6)")
    p.add_argument("--steps", type=int, default=1200,
                   help="Number of integration steps (default 1200)")
    p.add_argument("--sample-every", type=int, default=10,
                   help="Store COM point every N steps (default 10)")
    p.add_argument("--softening", type=float, default=1.0,
                   help="Softening length in kpc (default 1.0)")
    p.add_argument("--theta", type=float, default=0.5,
                   help="Barnes-Hut opening angle (default 0.5)")
    p.add_argument("--method", type=str, default="auto",
                   choices=["auto", "direct", "bh"],
                   help="Force method (default auto)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (run i uses seed+i)")
    p.add_argument("--output", type=str, default="trajectories_100.png",
                   help="Output image path (default trajectories_100.png)")
    p.add_argument("--dpi", type=int, default=180,
                   help="Output image DPI (default 180)")
    return p.parse_args()


def run_one_simulation(
    N: int,
    dt: float,
    steps: int,
    sample_every: int,
    softening: float,
    theta: float,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one simulation and return sampled COM trajectories.

    Returns
    -------
    t_gyr : (S,) sampled times in Gyr
    mw_xy : (S,2) Milky Way COM trajectory in x-y
    m31_xy: (S,2) Andromeda COM trajectory in x-y
    """
    pos, vel, mass = build_initial_conditions(N, seed=seed)
    n_total = pos.shape[0]
    n_half = n_total // 2

    use_direct = (
        method == "direct"
        or (method == "auto" and n_total <= 3000)
    )

    if use_direct:
        acc = compute_accelerations_direct(pos, mass, softening=softening)
    else:
        acc = compute_accelerations(pos, mass, softening=softening, theta=theta)

    n_samples = steps // sample_every + 1
    t_gyr = np.empty(n_samples, dtype=np.float64)
    mw_xy = np.empty((n_samples, 2), dtype=np.float64)
    m31_xy = np.empty((n_samples, 2), dtype=np.float64)

    # t = 0 sample
    sample_idx = 0
    t_gyr[sample_idx] = 0.0
    mw_xy[sample_idx] = pos[:n_half, :2].mean(axis=0)
    m31_xy[sample_idx] = pos[n_half:, :2].mean(axis=0)
    sample_idx += 1

    t_sim = 0.0
    for step in range(1, steps + 1):
        if use_direct:
            pos, vel, acc = leapfrog_step_direct(
                pos, vel, acc, mass,
                dt=dt,
                softening=softening,
            )
        else:
            pos, vel, acc = leapfrog_step(
                pos, vel, acc, mass,
                dt=dt,
                softening=softening,
                theta=theta,
            )

        t_sim += dt

        if step % sample_every == 0:
            t_gyr[sample_idx] = t_sim / 1e9
            mw_xy[sample_idx] = pos[:n_half, :2].mean(axis=0)
            m31_xy[sample_idx] = pos[n_half:, :2].mean(axis=0)
            sample_idx += 1

    return t_gyr, mw_xy, m31_xy


def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("Milkomeda Monte Carlo Center Trajectories")
    print("=" * 64)
    print(f"runs         : {args.runs}")
    print(f"N/galaxy     : {args.N}")
    print(f"steps        : {args.steps}")
    print(f"dt (yr)      : {args.dt:.2e}")
    print(f"duration(Gyr): {args.steps * args.dt / 1e9:.2f}")
    print(f"sample every : {args.sample_every}")
    print(f"method       : {args.method}")
    print(f"output       : {args.output}")
    print("=" * 64)

    all_mw = []
    all_m31 = []
    t_ref = None

    t0 = time.perf_counter()
    for i in tqdm(range(args.runs), desc="Simulations", ncols=80):
        seed_i = args.seed + i
        t_gyr, mw_xy, m31_xy = run_one_simulation(
            N=args.N,
            dt=args.dt,
            steps=args.steps,
            sample_every=args.sample_every,
            softening=args.softening,
            theta=args.theta,
            method=args.method,
            seed=seed_i,
        )
        if t_ref is None:
            t_ref = t_gyr
        all_mw.append(mw_xy)
        all_m31.append(m31_xy)

    all_mw = np.asarray(all_mw)    # (runs, S, 2)
    all_m31 = np.asarray(all_m31)  # (runs, S, 2)

    # Mean trajectories across all runs
    mw_mean = all_mw.mean(axis=0)
    m31_mean = all_m31.mean(axis=0)

    # Plot all trajectories in one figure
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="black")
    ax.set_facecolor("black")

    # Individual runs (transparent)
    for i in range(args.runs):
        ax.plot(all_mw[i, :, 0], all_mw[i, :, 1], color="#6ec6ff", alpha=0.16, lw=0.9)
        ax.plot(all_m31[i, :, 0], all_m31[i, :, 1], color="#ff9f59", alpha=0.16, lw=0.9)

    # Mean runs (emphasized)
    ax.plot(mw_mean[:, 0], mw_mean[:, 1], color="#c9ecff", lw=2.5, label="Milky Way center (mean)")
    ax.plot(m31_mean[:, 0], m31_mean[:, 1], color="#ffd2b0", lw=2.5, label="Andromeda center (mean)")

    # Start/end markers for mean trajectories
    ax.scatter(mw_mean[0, 0], mw_mean[0, 1], color="#55ddff", s=46, edgecolors="white", linewidths=0.5)
    ax.scatter(m31_mean[0, 0], m31_mean[0, 1], color="#ff7f3f", s=46, edgecolors="white", linewidths=0.5)
    ax.scatter(mw_mean[-1, 0], mw_mean[-1, 1], color="#55ddff", s=24)
    ax.scatter(m31_mean[-1, 0], m31_mean[-1, 1], color="#ff7f3f", s=24)

    ax.text(mw_mean[0, 0], mw_mean[0, 1], "  MW start", color="#bfe8ff", fontsize=9)
    ax.text(m31_mean[0, 0], m31_mean[0, 1], "  M31 start", color="#ffd5bf", fontsize=9)

    ax.set_title(
        f"Milkomeda: {args.runs} Simulated Center Trajectories (x-y projection)",
        color="white",
        fontsize=12,
    )
    ax.set_xlabel("x (kpc)", color="white")
    ax.set_ylabel("y (kpc)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax.legend(facecolor="#111111", edgecolor="white", labelcolor="white", fontsize=9)
    ax.grid(color="#333333", alpha=0.35, linestyle="--", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, facecolor="black")
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    print(f"Saved trajectory ensemble figure to '{args.output}'")
    print(f"Completed in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
