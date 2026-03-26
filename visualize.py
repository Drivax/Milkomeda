"""
visualize.py — Render the galaxy collision simulation as an MP4 or GIF animation.

Usage:
    python visualize.py --input output.h5 --fps 30 --colormap plasma
    python visualize.py --input output.h5 --output milkomeda.gif --fps 15

Output format is auto-detected from the file extension:
  .gif  → Pillow writer (no ffmpeg required)
  .mp4  → FFMpeg writer (ffmpeg must be installed)

Requires: matplotlib, h5py, numpy, Pillow (for GIF), ffmpeg (for MP4)
"""

import argparse
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

matplotlib.use("Agg")  # headless rendering


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milkomeda — Visualize galaxy collision simulation"
    )
    p.add_argument("--input", type=str, default="output.h5",
                   help="Input HDF5 file from simulate.py (default output.h5)")
    p.add_argument("--output", type=str, default="milkomeda.mp4",
                   help="Output MP4 file (default milkomeda.mp4)")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second in the output video (default 30)")
    p.add_argument("--colormap", type=str, default="plasma",
                   help="Matplotlib colormap for particle density (default plasma)")
    p.add_argument("--dpi", type=int, default=150,
                   help="DPI of output frames (default 150)")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Particle alpha (transparency) value (default 0.3)")
    p.add_argument("--size", type=float, default=0.2,
                   help="Particle marker size (default 0.2)")
    p.add_argument("--view", type=str, default="xy",
                   choices=["xy", "xz", "yz"],
                   help="Projection plane (default xy)")
    p.add_argument("--energy", action="store_true",
                   help="Plot energy conservation panel alongside animation")
    p.add_argument("--loop", type=int, default=0,
                   help="GIF loop count (0 = infinite, default 0)")
    p.add_argument("--hold-initial", type=int, default=24,
                   help="Number of extra frames to hold the initial state (default 24)")
    p.add_argument("--show-trajectories", action="store_true",
                   help="Overlay galaxy center-of-mass trajectories")
    p.add_argument("--traj-width", type=float, default=1.2,
                   help="Line width for trajectory curves (default 1.2)")
    p.add_argument("--density-colors", action="store_true",
                   help="Use per-particle density colormaps instead of fixed galaxy colors")
    p.add_argument("--show-centers", action="store_true", default=True,
                   help="Show center-of-mass markers for both galaxies")
    p.add_argument("--full-extent", action="store_true",
                   help="Use full particle extent for axis limits (can make initial separation look small)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_indices(view: str) -> tuple[int, int, str, str]:
    mapping = {
        "xy": (0, 1, "x (kpc)", "y (kpc)"),
        "xz": (0, 2, "x (kpc)", "z (kpc)"),
        "yz": (1, 2, "y (kpc)", "z (kpc)"),
    }
    return mapping[view]


def _density_color(pos_2d: np.ndarray, cmap_name: str, n_bins: int = 256):
    """Assign a colour to each particle based on local projected density."""
    xmin, xmax = pos_2d[:, 0].min(), pos_2d[:, 0].max()
    ymin, ymax = pos_2d[:, 1].min(), pos_2d[:, 1].max()
    H, xedges, yedges = np.histogram2d(
        pos_2d[:, 0], pos_2d[:, 1],
        bins=n_bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    H = np.log1p(H)
    # Map each particle to its bin
    xi = np.clip(
        np.searchsorted(xedges, pos_2d[:, 0]) - 1, 0, n_bins - 1)
    yi = np.clip(
        np.searchsorted(yedges, pos_2d[:, 1]) - 1, 0, n_bins - 1)
    density = H[xi, yi]
    density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    return cmap(density_norm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file '{args.input}' not found. "
                                "Run simulate.py first.")

    print(f"Loading '{args.input}'...")
    with h5py.File(args.input, "r") as hf:
        pos_all = hf["pos"][:]        # (n_snaps, N, 3) float32
        times = hf["time"][:]         # (n_snaps,)
        N_total = hf["mass"].shape[0]
        N_half = N_total // 2

        has_energy = "KE" in hf and args.energy
        if has_energy:
            KE_arr = hf["KE"][:]
            PE_arr = hf["PE"][:]
            E_arr = KE_arr + PE_arr

    n_snaps = pos_all.shape[0]
    ax_i, ax_j, xlabel, ylabel = _axis_indices(args.view)

    # Centers of mass over time (projected to selected view) for trajectory overlay
    com_mw = pos_all[:, :N_half, :].mean(axis=1)
    com_and = pos_all[:, N_half:, :].mean(axis=1)

    print(f"  Snapshots : {n_snaps}")
    print(f"  Particles : {N_total:,}")
    print(f"  Time span : {times[-1]/1e9:.2f} Gyr")

    # --- Figure layout ---
    if has_energy:
        fig, (ax_sim, ax_en) = plt.subplots(
            1, 2, figsize=(14, 7),
            gridspec_kw={"width_ratios": [2, 1]},
            facecolor="black",
        )
    else:
        fig, ax_sim = plt.subplots(1, 1, figsize=(9, 9), facecolor="black")

    ax_sim.set_facecolor("black")
    ax_sim.set_xlabel(xlabel, color="white")
    ax_sim.set_ylabel(ylabel, color="white")
    ax_sim.tick_params(colors="white")
    for spine in ax_sim.spines.values():
        spine.set_edgecolor("white")

    # Axis limits: default to COM-trajectory framing so both galaxies are clearly
    # separated at t=0 and their approach/collision remains visible.
    if args.full_extent:
        all_x = pos_all[:, :, ax_i]
        all_y = pos_all[:, :, ax_j]
        p1, p99_x = np.percentile(all_x, [1, 99])
        p1_y, p99_y = np.percentile(all_y, [1, 99])
        pad = 50
        xlim = (p1 - pad, p99_x + pad)
        ylim = (p1_y - pad, p99_y + pad)
    else:
        com_x = np.concatenate([com_mw[:, ax_i], com_and[:, ax_i]])
        com_y = np.concatenate([com_mw[:, ax_j], com_and[:, ax_j]])
        p1, p99_x = np.percentile(com_x, [1, 99])
        p1_y, p99_y = np.percentile(com_y, [1, 99])
        pad = 180
        xlim = (p1 - pad, p99_x + pad)
        ylim = (p1_y - pad, p99_y + pad)
    ax_sim.set_xlim(*xlim)
    ax_sim.set_ylim(*ylim)

    # Galaxy 1 (MW) scatter
    scat_mw = ax_sim.scatter([], [], s=args.size, alpha=args.alpha,
                              linewidths=0, rasterized=True)
    scat_and = ax_sim.scatter([], [], s=args.size, alpha=args.alpha,
                               linewidths=0, rasterized=True)

    traj_mw, = ax_sim.plot([], [], color="#8cc8ff", lw=args.traj_width, alpha=0.9)
    traj_and, = ax_sim.plot([], [], color="#ff9a52", lw=args.traj_width, alpha=0.9)

    center_mw = ax_sim.scatter([], [], s=36, color="#8cc8ff", edgecolors="white", linewidths=0.4)
    center_and = ax_sim.scatter([], [], s=36, color="#ff9a52", edgecolors="white", linewidths=0.4)
    label_mw = ax_sim.text(0, 0, "MW", color="#b8deff", fontsize=8, ha="left", va="bottom")
    label_and = ax_sim.text(0, 0, "M31", color="#ffd0ad", fontsize=8, ha="left", va="bottom")

    title = ax_sim.set_title("", color="white", fontsize=10)

    if has_energy:
        ax_en.set_facecolor("#111111")
        ax_en.set_xlabel("Time (Gyr)", color="white")
        ax_en.set_ylabel("Energy (M☉ kpc² yr⁻²)", color="white")
        ax_en.tick_params(colors="white")
        for spine in ax_en.spines.values():
            spine.set_edgecolor("white")
        ax_en.set_title("Energy Conservation", color="white", fontsize=9)
        ax_en.plot(times / 1e9, E_arr, color="cyan", lw=0.8, label="E_total")
        ax_en.plot(times / 1e9, KE_arr, color="orange", lw=0.6, alpha=0.7, label="KE")
        ax_en.plot(times / 1e9, PE_arr, color="lime", lw=0.6, alpha=0.7, label="PE")
        ax_en.legend(fontsize=7, labelcolor="white",
                     facecolor="#1a1a1a", edgecolor="white")
        vline = ax_en.axvline(0, color="white", lw=0.8, ls="--")

    fig.tight_layout(pad=1.0)

    # --- Animation update function ---
    n_frames = n_snaps + max(0, args.hold_initial)

    def update(frame: int):
        sim_idx = max(0, frame - max(0, args.hold_initial))
        snap = pos_all[sim_idx]
        x_mw  = snap[:N_half, ax_i]
        y_mw  = snap[:N_half, ax_j]
        x_and = snap[N_half:, ax_i]
        y_and = snap[N_half:, ax_j]

        pos_mw2 = np.column_stack([x_mw, y_mw])
        pos_and2 = np.column_stack([x_and, y_and])

        scat_mw.set_offsets(pos_mw2)
        scat_and.set_offsets(pos_and2)
        if args.density_colors:
            c_mw = _density_color(pos_mw2, "Blues")
            c_and = _density_color(pos_and2, args.colormap)
            scat_mw.set_color(c_mw)
            scat_and.set_color(c_and)
        else:
            scat_mw.set_color("#6ec6ff")
            scat_and.set_color("#ff9f59")

        mw_cx, mw_cy = com_mw[sim_idx, ax_i], com_mw[sim_idx, ax_j]
        and_cx, and_cy = com_and[sim_idx, ax_i], com_and[sim_idx, ax_j]
        if args.show_centers:
            center_mw.set_offsets([[mw_cx, mw_cy]])
            center_and.set_offsets([[and_cx, and_cy]])
            label_mw.set_position((mw_cx + 8, mw_cy + 8))
            label_and.set_position((and_cx + 8, and_cy + 8))
        else:
            center_mw.set_offsets(np.empty((0, 2)))
            center_and.set_offsets(np.empty((0, 2)))
            label_mw.set_position((1e9, 1e9))
            label_and.set_position((1e9, 1e9))

        t_gyr = times[sim_idx] / 1e9
        if args.show_trajectories:
            traj_mw.set_data(com_mw[:sim_idx + 1, ax_i], com_mw[:sim_idx + 1, ax_j])
            traj_and.set_data(com_and[:sim_idx + 1, ax_i], com_and[:sim_idx + 1, ax_j])
        else:
            traj_mw.set_data([], [])
            traj_and.set_data([], [])

        state_text = "initial state" if frame < max(0, args.hold_initial) else "evolution"
        title.set_text(
            f"Milkomeda  |  t = {t_gyr:.3f} Gyr  |  {state_text}  |  frame {frame+1}/{n_frames}"
        )

        artists = [
            scat_mw, scat_and,
            traj_mw, traj_and,
            center_mw, center_and,
            label_mw, label_and,
            title,
        ]
        if has_energy:
            vline.set_xdata([t_gyr, t_gyr])
            artists.append(vline)
        return artists

    print(f"Rendering {n_frames} frames at {args.fps} fps...")
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 / args.fps,
        blit=True,
    )

    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".gif":
        writer = animation.PillowWriter(
            fps=args.fps,
            metadata={"title": "Milkomeda Galaxy Collision"},
        )
        save_kwargs = {
            "savefig_kwargs": {"facecolor": "black"},
            "writer": writer,
            "dpi": args.dpi,
        }
        # Pass loop count via extra PillowWriter kwarg if supported
        ani.save(args.output, **save_kwargs)
        # Re-open GIF to bake in loop count using Pillow directly
        try:
            from PIL import Image
            frames_pil = []
            img = Image.open(args.output)
            try:
                while True:
                    frames_pil.append(img.copy().convert("RGBA"))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            if frames_pil:
                frames_pil[0].save(
                    args.output,
                    save_all=True,
                    append_images=frames_pil[1:],
                    loop=args.loop,
                    optimize=False,
                )
        except Exception:
            pass  # non-critical; file already saved
    else:
        writer = animation.FFMpegWriter(
            fps=args.fps,
            metadata={"title": "Milkomeda Galaxy Collision"},
            bitrate=4000,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        ani.save(args.output, writer=writer, dpi=args.dpi,
                 savefig_kwargs={"facecolor": "black"})

    print(f"Animation saved to '{args.output}'.")


if __name__ == "__main__":
    main()
