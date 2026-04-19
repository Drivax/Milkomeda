"""Interactive Streamlit app for Milkomeda simulation outputs.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def _axis_indices(view: str) -> tuple[int, int, str, str]:
    mapping = {
        "xy": (0, 1, "x (kpc)", "y (kpc)"),
        "xz": (0, 2, "x (kpc)", "z (kpc)"),
        "yz": (1, 2, "y (kpc)", "z (kpc)"),
    }
    return mapping[view]


@st.cache_data(show_spinner=False)
def find_h5_files(root_dir: str) -> list[str]:
    root = Path(root_dir)
    return sorted(str(p) for p in root.glob("*.h5"))


@st.cache_data(show_spinner=True)
def load_simulation(path: str) -> dict:
    with h5py.File(path, "r") as hf:
        data = {
            "pos": hf["pos"][:],
            "time": hf["time"][:],
            "mass": hf["mass"][:],
            "metadata": dict(hf.get("metadata", {}).attrs.items()) if "metadata" in hf else {},
            "has_energy": ("KE" in hf and "PE" in hf),
        }
        if "vel" in hf:
            data["vel"] = hf["vel"][:]
        if data["has_energy"]:
            data["KE"] = hf["KE"][:]
            data["PE"] = hf["PE"][:]
    return data


@st.cache_data(show_spinner=False)
def compute_com_and_separation(pos: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = mass.shape[0]
    n_half = n_total // 2

    mw_mass = mass[:n_half]
    and_mass = mass[n_half:]

    mw_norm = mw_mass / (mw_mass.sum() + 1e-300)
    and_norm = and_mass / (and_mass.sum() + 1e-300)

    com_mw = (pos[:, :n_half, :] * mw_norm[None, :, None]).sum(axis=1)
    com_and = (pos[:, n_half:, :] * and_norm[None, :, None]).sum(axis=1)
    separation = np.linalg.norm(com_and - com_mw, axis=1)
    return com_mw, com_and, separation


def _format_metric_number(value: float, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _time_scale_config(unit: str) -> tuple[float, str]:
    mapping = {
        "yr": (1.0, "yr"),
        "Myr": (1e6, "Myr"),
        "Gyr": (1e9, "Gyr"),
    }
    return mapping[unit]


def main() -> None:
    st.set_page_config(
        page_title="Milkomeda Explorer",
        page_icon="*",
        layout="wide",
    )

    st.title("Milkomeda Simulation Explorer")
    st.caption("Interactive viewer for HDF5 outputs from simulate.py")

    h5_files = find_h5_files(".")
    if not h5_files:
        st.error("No .h5 files found in the current folder.")
        st.info("Generate one with simulate.py first, then re-run this app.")
        return

    with st.sidebar:
        st.header("Data")
        selected_path = st.selectbox("Simulation file", h5_files, index=0)

    sim = load_simulation(selected_path)
    pos = sim["pos"]
    times = sim["time"]
    mass = sim["mass"]
    meta = sim["metadata"]

    n_snaps, n_total, _ = pos.shape
    n_half = n_total // 2
    com_mw, com_and, separation = compute_com_and_separation(pos, mass)

    with st.sidebar:
        st.header("View")
        projection = st.selectbox("Projection", ["xy", "xz", "yz"], index=0)
        time_unit = st.selectbox("Time scale", ["yr", "Myr", "Gyr"], index=2)
        snap_idx = st.slider("Snapshot index", 0, n_snaps - 1, value=0)
        max_points = st.slider(
            "Max particles to draw per galaxy",
            min_value=200,
            max_value=max(200, n_half),
            value=min(5000, n_half),
            step=200,
        )
        show_trajectories = st.checkbox("Show center trajectories", value=True)
        show_centers = st.checkbox("Show centers", value=True)
        lock_aspect = st.checkbox("Equal axis scale", value=True)
        framing = st.radio("Axis framing", ["COM-based", "Full extent"], index=0)

    time_scale, time_label = _time_scale_config(time_unit)
    times_scaled = times / time_scale
    current_time_scaled = float(times_scaled[snap_idx])

    min_sep_idx = int(np.argmin(separation))
    min_sep_kpc = float(separation[min_sep_idx])
    min_sep_time_scaled = float(times_scaled[min_sep_idx])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Particles", f"{n_total:,}")
    col2.metric("Snapshots", f"{n_snaps:,}")
    col3.metric(f"Current time ({time_label})", _format_metric_number(current_time_scaled, 3))
    col4.metric("Min COM separation (kpc)", _format_metric_number(min_sep_kpc, 2))

    info_cols = st.columns(3)
    info_cols[0].write(f"**File:** {selected_path}")
    info_cols[1].write(f"**Min separation time:** {min_sep_time_scaled:.3f} {time_label}")
    first_close = meta.get("first_close_approach_time_gyr", None)
    if isinstance(first_close, (int, float)) and first_close >= 0:
        first_close_scaled = float((first_close * 1e9) / time_scale)
        info_cols[2].write(f"**First close approach:** {first_close_scaled:.3f} {time_label}")
    else:
        info_cols[2].write("**First close approach:** n/a")

    ax_i, ax_j, xlabel, ylabel = _axis_indices(projection)

    snap = pos[snap_idx]
    mw = snap[:n_half]
    andr = snap[n_half:]

    if max_points < n_half:
        idx_mw = np.linspace(0, n_half - 1, max_points, dtype=int)
        idx_and = np.linspace(0, n_half - 1, max_points, dtype=int)
        mw = mw[idx_mw]
        andr = andr[idx_and]

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    ax.scatter(
        mw[:, ax_i],
        mw[:, ax_j],
        s=2.5,
        alpha=0.55,
        c="#6ec6ff",
        linewidths=0,
        label="Milky Way",
    )
    ax.scatter(
        andr[:, ax_i],
        andr[:, ax_j],
        s=2.5,
        alpha=0.55,
        c="#ff9f59",
        linewidths=0,
        label="Andromeda",
    )

    if show_trajectories:
        ax.plot(
            com_mw[: snap_idx + 1, ax_i],
            com_mw[: snap_idx + 1, ax_j],
            lw=1.4,
            c="#9ad6ff",
            alpha=0.95,
        )
        ax.plot(
            com_and[: snap_idx + 1, ax_i],
            com_and[: snap_idx + 1, ax_j],
            lw=1.4,
            c="#ffc08c",
            alpha=0.95,
        )

    if show_centers:
        ax.scatter(
            [com_mw[snap_idx, ax_i]],
            [com_mw[snap_idx, ax_j]],
            s=36,
            c="#9ad6ff",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
        ax.scatter(
            [com_and[snap_idx, ax_i]],
            [com_and[snap_idx, ax_j]],
            s=36,
            c="#ffc08c",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    if framing == "Full extent":
        all_x = pos[:, :, ax_i]
        all_y = pos[:, :, ax_j]
        x_low, x_high = np.percentile(all_x, [1, 99])
        y_low, y_high = np.percentile(all_y, [1, 99])
        pad = 50
        ax.set_xlim(x_low - pad, x_high + pad)
        ax.set_ylim(y_low - pad, y_high + pad)
    else:
        com_x = np.concatenate([com_mw[:, ax_i], com_and[:, ax_i]])
        com_y = np.concatenate([com_mw[:, ax_j], com_and[:, ax_j]])
        x_low, x_high = np.percentile(com_x, [1, 99])
        y_low, y_high = np.percentile(com_y, [1, 99])
        pad = 180
        ax.set_xlim(x_low - pad, x_high + pad)
        ax.set_ylim(y_low - pad, y_high + pad)

    if lock_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(loc="upper right", facecolor="#1b1f2a", edgecolor="white", labelcolor="white")
    ax.set_title(f"Milkomeda snapshot {snap_idx + 1}/{n_snaps} at t={current_time_scaled:.3f} {time_label}", color="white")

    st.pyplot(fig, clear_figure=True)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Center Separation vs Time")
        fig_sep, ax_sep = plt.subplots(figsize=(7.5, 3.3))
        ax_sep.plot(times_scaled, separation, c="#34a0a4", lw=1.8)
        ax_sep.axvline(current_time_scaled, c="#ef476f", lw=1.1, ls="--")
        ax_sep.set_xlabel(f"Time ({time_label})")
        ax_sep.set_ylabel("Separation (kpc)")
        ax_sep.grid(alpha=0.25)
        st.pyplot(fig_sep, clear_figure=True)

    with chart_right:
        st.subheader("Simulation Metadata")
        if meta:
            meta_rows = []
            for k in sorted(meta.keys()):
                v = meta[k]
                if isinstance(v, np.generic):
                    v = v.item()
                meta_rows.append({"key": str(k), "value": str(v)})
            st.dataframe(meta_rows, use_container_width=True, height=240)
        else:
            st.info("No metadata group found in this file.")

    if sim["has_energy"]:
        st.subheader("Energy Diagnostics")
        ke = sim["KE"]
        pe = sim["PE"]
        e_tot = ke + pe
        fig_e, ax_e = plt.subplots(figsize=(11, 3.5))
        ax_e.plot(times_scaled, e_tot, label="E total", c="#06d6a0", lw=1.5)
        ax_e.plot(times_scaled, ke, label="KE", c="#ffd166", lw=1.0, alpha=0.9)
        ax_e.plot(times_scaled, pe, label="PE", c="#118ab2", lw=1.0, alpha=0.9)
        ax_e.axvline(current_time_scaled, c="#ef476f", lw=1.1, ls="--")
        ax_e.set_xlabel(f"Time ({time_label})")
        ax_e.set_ylabel("Energy")
        ax_e.grid(alpha=0.2)
        ax_e.legend()
        st.pyplot(fig_e, clear_figure=True)


if __name__ == "__main__":
    main()
