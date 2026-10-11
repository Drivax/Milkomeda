# Local audit

The KDK leapfrog update and softened direct pair energy were coherent and retained.
The original scripts duplicated integration, defaults and COM calculations.
Streamlit was a viewer and eagerly loaded all particle arrays.

Corrections:
- NFW radius sampling used density without the r² volume factor; it now inverts
  enclosed mass. This changes initial conditions, so old plots are not comparable.
- Each finite galaxy is recentered in position and velocity before applying the
  requested relative state. Previously sampling offsets changed actual COM distance/speed.
- Galaxy masses, scales, orientation and velocity dispersion now come from one config.
- Barnes–Hut must open a cell containing the target particle, avoiding self-mass
  contributions through a cell aggregate.
- The final state is retained when the save stride does not divide step count.
- Null indicates unavailable energy validation or no threshold crossing; it is not zero.
- Shared diagnostics use mass-weighted COM and a nonzero momentum normalization.
- The old claim of an exact Miyamoto–Nagai sampling/equilibrium was removed:
  the sampler and velocity prescription are approximations.

Original force solvers, leapfrog and useful galaxy sampling routines remain.
Repeated simulation loops and the viewer were replaced by shared simulation,
plotting and configuration functions. Old outputs remain only in ignored legacy/.
