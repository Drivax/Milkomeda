# 🌌 Galaxy Collision Simulator — Milky Way × Andromeda


> A naive N-body gravitational simulation of the future collision between the Milky Way and the Andromeda galaxy (M31), grounded in real observational data and classical mechanics.

***

##  Motivation

In approximately **4.5 billion years**, the Milky Way and Andromeda will collide and eventually merge into a single elliptical galaxy, sometimes nicknamed **"Milkomeda"**.

This project started as a personal question: *can we reproduce this cosmic ballet using nothing but Newton's laws, real astrometric data, and a few thousand lines of Python?*

***

##  The Physics

### Gravity — The Only Rule That Matters

At its core, this simulation relies on **Newton's Law of Universal Gravitation**:

```
F = G * (m1 * m2) / r²
```

Where:

* `G = 6.674 × 10⁻¹¹ N·m²·kg⁻²` is the gravitational constant

* `m1`, `m2` are the masses of two interacting bodies

* `r` is the distance between them

Every star (or dark matter particle) in the simulation feels the gravitational pull of every other particle. This is the **N-body problem**.

### The N-Body Problem

For N particles, the acceleration of particle `i` is:

```
a_i = G * Σ_{j ≠ i} [ m_j * (r_j - r_i) / |r_j - r_i|³ ]
```

This is computationally expensive: naively, it scales as **O(N²)**. For N = 10,000 particles, that's 100 million force calculations per timestep. We address this with the **Barnes-Hut algorithm** (see below).

### Softening Length

To avoid singularities when two particles get very close (division by near-zero), we introduce a **softening parameter ε**:

```
a_i = G * Σ_{j ≠ i} [ m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2) ]
```

Think of ε as a "blur radius" — it prevents unphysical infinite accelerations when particles pass through each other. Typical values: `ε ~ 0.1 kpc`.

***

##  The Mathematics

### Numerical Integration — Leapfrog Scheme

We use the **Leapfrog (Störmer-Verlet) integrator**, which is symplectic — meaning it conserves energy over long timescales far better than a naive Euler method.

The update equations are:

```
v_{i+1/2} = v_{i-1/2} + a_i * Δt          (kick)
x_{i+1}   = x_i + v_{i+1/2} * Δt          (drift)
a_{i+1}   = f(x_{i+1})                     (update forces)
v_{i+1}   = v_{i+1/2} + a_{i+1} * Δt/2    (kick again)
```

Why leapfrog? Because galaxies take billions of years to merge. A non-symplectic integrator would slowly bleed or inject energy into the system, making the simulation drift into nonsense. Leapfrog keeps the total energy oscillating around a constant — physically honest.

### Barnes-Hut Tree Algorithm — O(N log N)

Instead of computing all N² pairwise interactions, we use an **octree** (3D) to group distant particles into single "super-particles":

* If a cluster of particles is **far enough away** (ratio `s/d < θ`, where `s` is the cell size and `d` is the distance), treat the whole cluster as one body located at its **center of mass**.

* The threshold `θ ≈ 0.5` is the opening angle parameter — lower = more accurate, higher = faster.

This reduces complexity from **O(N²)** to **O(N log N)**, making simulations with tens of thousands of particles tractable on a laptop.

***

##  Initial Conditions — Real Data

We use actual observational measurements:

| Parameter                | Milky Way     | Andromeda (M31)                       |
| ------------------------ | ------------- | ------------------------------------- |
| Stellar Mass             | \~6 × 10¹⁰ M☉ | \~10¹¹ M☉                             |
| Dark Matter Halo Mass    | \~10¹² M☉     | \~1.5 × 10¹² M☉                       |
| Distance                 | —             | \~785 kpc                             |
| Relative radial velocity | —             | -110 km/s (approaching)               |
| Transverse velocity      | —             | \~17 km/s (van der Marel et al. 2012) |
| Disk inclination (MW)    | —             | \~77°                                 |

The **transverse velocity** is critical: it determines whether the galaxies collide head-on or in a glancing blow. The low transverse velocity measured by HST strongly suggests a near head-on collision.

Each galaxy is initialized as a **Hernquist profile** for the bulge and a **Miyamoto-Nagai disk** for the stellar disk, embedded in a **NFW dark matter halo**.

### NFW Dark Matter Halo Profile

```
ρ(r) = ρ_s / [ (r/r_s) * (1 + r/r_s)² ]
```

Where `ρ_s` is a characteristic density and `r_s` is the scale radius. Dark matter dominates the mass budget and drives the large-scale dynamics of the merger.

### Hernquist Bulge Profile

```
ρ(r) = M_b / (2π) * a / [ r * (r + a)³ ]
```

Where `a` is the scale length and `M_b` is the bulge mass.

***

##  Getting Started

### Requirements

```bash
pip install numpy scipy matplotlib astropy tqdm h5py
```

### Run the simulation

```bash
git clone https://github.com/yourname/galaxy-collision-sim
cd galaxy-collision-sim
python simulate.py --N 5000 --dt 1e6 --steps 2000 --theta 0.5
```

### Parameters

| Argument      | Description                    | Default     |
| ------------- | ------------------------------ | ----------- |
| `--N`         | Number of particles per galaxy | 5000        |
| `--dt`        | Timestep in years              | 1e6         |
| `--steps`     | Number of integration steps    | 2000        |
| `--theta`     | Barnes-Hut opening angle       | 0.5         |
| `--softening` | Softening length in kpc        | 0.1         |
| `--output`    | Output HDF5 file               | `output.h5` |

***

##  Outputs & Visualization

The simulation saves particle positions and velocities at each snapshot into an HDF5 file. Visualization is handled by `visualize.py`:

```bash
python visualize.py --input output.h5 --fps 30 --colormap plasma
```

This generates an MP4 animation showing the full merger sequence — from first approach, through first pass, multiple oscillations, and final coalescence into an elliptical remnant.

***

##  Validation

To validate the simulation, we check:

* **Energy conservation**: total energy (kinetic + potential) should remain within \~1% over the full run with leapfrog.

* **Angular momentum conservation**: should be conserved to machine precision.

* **Virial theorem**: for an isolated galaxy in equilibrium, `2 * KE + PE ≈ 0`.

* **Comparison with literature**: merger timescale (\~5–6 Gyr from now) and morphology of tidal tails match published N-body results (Cox & Loeb 2008).

***




<br />
