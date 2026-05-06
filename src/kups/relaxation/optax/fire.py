# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""FIRE optimizers as composable Optax transforms.

Provides:
    - ``scale_by_fire``: original FIRE (Bitzek et al. 2006).
    - ``scale_by_fire2``: FIRE 2.0 (Guénolé et al. 2020) with optional
      ABC-FIRE bias correction (Echeverri Restrepo & Andric 2023).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array


class ScaleByFireState(NamedTuple):
    """State for scale_by_fire transform.

    Attributes:
        velocity: Velocity estimate (PyTree matching params).
        dt: Current adaptive timestep.
        alpha: Current velocity mixing parameter.
        n_pos: Count of consecutive positive power steps.
    """

    velocity: optax.Params
    dt: Array
    alpha: Array
    n_pos: Array


class ScaleByFire2State(NamedTuple):
    """State for scale_by_fire2 transform.

    Attributes:
        velocity: Velocity estimate (PyTree matching params).
        dt: Current adaptive timestep.
        alpha: Current velocity mixing parameter.
        n_pos: Number of steps since the last non-positive power event
            (used both for the ``n_min`` delay and for ABC-FIRE's bias
            correction exponent).
        n_total: Total number of update steps taken (for
            ``delaystep_start``).
    """

    velocity: optax.Params
    dt: Array
    alpha: Array
    n_pos: Array
    n_total: Array


def scale_by_fire(
    dt_start: float = 0.1,
    dt_max: float | None = None,
    dt_min: float | None = None,
    max_step: float | None = 0.2,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    n_min: int = 5,
) -> optax.GradientTransformation:
    """FIRE (Fast Inertial Relaxation Engine) optimizer.

    Composable Optax transform implementing the FIRE algorithm for
    structure relaxation. Can be chained with other transforms.

    .. note::

        This is the original FIRE 1.0 (Bitzek 2006). For most
        production relaxations prefer :func:`scale_by_fire2`, which
        Guénolé et al. 2020 (Fig. 4–6) report converges in ~1.5–3×
        fewer force calls on Lennard-Jones, EAM and Tersoff
        benchmarks. ABC-FIRE (``use_abc=True``, Echeverri Restrepo &
        Andric 2023, Fig. 2–3) is typically a further ~10–40%
        faster, but takes more aggressive steps and is correspondingly
        more prone to diverging on poorly conditioned or noisy
        landscapes — enable it only after a plain FIRE 2.0 run is
        known to be stable. FIRE 1.0 remains useful as a well-tested
        baseline and for comparison with legacy results.

    Args:
        dt_start: Initial timestep.
        dt_max: Maximum timestep. Defaults to 10 * dt_start.
        dt_min: Minimum timestep. Defaults to dt_start * 1e-4.
        max_step: Maximum step size (clips position updates). Defaults to 0.2 Å.
            Set to None to disable clipping.
        f_inc: Factor to increase dt when making progress.
        f_dec: Factor to decrease dt on bad step.
        alpha_start: Initial velocity mixing parameter.
        f_alpha: Factor to decay alpha when making progress.
        n_min: Minimum positive power steps before increasing dt.

    Returns:
        Optax GradientTransformation implementing FIRE.

    Reference:
        Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
    """
    if dt_max is None:
        dt_max = 10.0 * dt_start
    if dt_min is None:
        dt_min = dt_start * 1e-4

    def init_fn(params: optax.Params) -> ScaleByFireState:
        return ScaleByFireState(
            velocity=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt_start),
            alpha=jnp.array(alpha_start),
            n_pos=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates,
        state: ScaleByFireState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByFireState]:
        del params

        # F = -gradient (FIRE uses forces, pointing downhill)
        forces = jax.tree.map(lambda g: -g, updates)

        # Update velocity: v = v + dt * F
        velocity = jax.tree.map(lambda v, f: v + state.dt * f, state.velocity, forces)

        # Compute power: P = F · v (positive when moving downhill)
        power = optax.tree_utils.tree_vdot(forces, velocity)
        positive_power = power > 0.0  # type: ignore

        # Velocity mixing: v = (1-α)v + α|v|F̂
        v_norm = optax.tree_utils.tree_norm(velocity)
        f_norm = optax.tree_utils.tree_norm(forces)
        safe_f_norm = jnp.maximum(f_norm, 1e-10)

        mixed_velocity = jax.tree.map(
            lambda v, f: (1 - state.alpha) * v + state.alpha * v_norm * f / safe_f_norm,
            velocity,
            forces,
        )

        # Adaptive timestep and mixing parameter
        should_increase = jnp.logical_and(positive_power, state.n_pos >= n_min)

        new_dt = jnp.where(
            positive_power,
            jnp.where(should_increase, jnp.minimum(state.dt * f_inc, dt_max), state.dt),
            jnp.maximum(state.dt * f_dec, dt_min),
        )
        new_alpha = jnp.where(
            positive_power,
            jnp.where(should_increase, state.alpha * f_alpha, state.alpha),
            alpha_start,
        )
        new_n_pos = jnp.where(positive_power, state.n_pos + 1, 0)

        # If P > 0: use mixed velocity for next step and position update
        # If P <= 0: reset velocity to zero, no position update
        final_velocity = jax.tree.map(
            lambda v: jnp.where(positive_power, v, jnp.zeros_like(v)),
            mixed_velocity,
        )

        # Position update: step only when making progress (P > 0)
        position_updates = jax.tree.map(
            lambda v: jnp.where(positive_power, state.dt * v, jnp.zeros_like(v)),
            mixed_velocity,
        )

        # Clip position updates to max_step (prevents runaway steps)
        if max_step is not None:
            update_norm = optax.tree_utils.tree_norm(position_updates)
            scale = jnp.minimum(1.0, max_step / jnp.maximum(update_norm, 1e-10))
            position_updates = jax.tree.map(lambda u: u * scale, position_updates)

        return position_updates, ScaleByFireState(
            velocity=final_velocity, dt=new_dt, alpha=new_alpha, n_pos=new_n_pos
        )

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore[arg-type]


def scale_by_fire2(
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    dt_min: float = 2e-3,
    max_step: float | None = 0.1,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.25,
    f_alpha: float = 0.99,
    n_min: int = 20,
    use_abc: bool = False,
    halfstepback: bool = True,
    delaystep_start: bool = True,
) -> optax.GradientTransformation:
    """FIRE 2.0 (with optional ABC-FIRE) as a composable Optax transform.

    This implementation follows the LAMMPS ``min_fire.cpp`` Euler-implicit
    integrator (``eulerimplicit``) as closely as possible within the Optax
    gradient-transform framework (no MPI, no per-atom masses, no
    alternative integrators). The only intentional mismatch is the
    non-ABC ``P <= 0`` recovery path: LAMMPS can recompute forces after
    the half-step backtrack inside the same iteration, whereas a pure
    Optax transform only sees the incoming gradient once per step, so
    the recovery kick reuses that gradient.

    .. note::

        **When to pick which.** Guénolé et al. 2020 (Fig. 4–6) show
        FIRE 2.0 reaching the same convergence threshold in roughly
        1.5–3× fewer force evaluations than FIRE 1.0 across a wide
        range of EAM, LJ and Tersoff benchmarks. ABC-FIRE
        (``use_abc=True``, Echeverri Restrepo & Andric 2023, Fig. 2–3)
        is typically a further ~10–40% faster on dislocation and
        grain-boundary relaxations, with the largest gains in the
        first ~``n_min`` iterations after each ``P ≤ 0`` event.
        However, the bias-corrected mixing makes the early-step
        velocity update much larger than plain FIRE 2.0, so ABC-FIRE
        is more prone to overshoot and divergence on noisy or
        poorly conditioned potentials. **Recommended workflow:**
        always start with the more conservative FIRE 2.0
        (``use_abc=False``); only switch to ABC once that run is
        known to converge cleanly and you can tolerate the extra
        risk in exchange for the speed-up. Keep :func:`scale_by_fire`
        for cross-checking against legacy results.

    Key algorithmic steps per iteration (matching LAMMPS):

    1. Compute power ``P = F · v_old``.
    2. If ``P > 0``: compute mixing scales; if past delay, grow ``dt``
       and shrink ``alpha``.
    3. If ``P <= 0``: (optionally) half-step backtrack, reset velocity,
       shrink ``dt`` (LAMMPS-style: only if result ≥ ``dt_min``), reset
       ``alpha``.
    4. ``dmax`` limiting (non-ABC): compute ``dtv`` from ∞-norm of
       ``v_old`` (P > 0) or estimated ``dt·F`` (P ≤ 0).  ``dtv`` is
       used for both the velocity Euler kick and the position update.
    5. Euler-implicit kick: ``v += dtv · F``.
    6. Mixing (P > 0 only): ``v = scale1·v + scale2·F``.
    7. ABC per-component clip (P > 0 only).
    8. Position update: ``Δx = dtv · v`` (plus backtrack on P ≤ 0).

    When ``use_abc=True`` the velocity mixing is replaced by the
    Accelerated Bias-Corrected FIRE update (Echeverri Restrepo & Andric
    2023):

    .. math::

        v \\leftarrow \\frac{1}{1-(1-\\alpha)^N}
                     \\bigl[(1-\\alpha)\\,v + \\alpha\\,|v|\\,\\hat F\\bigr]

    where ``N`` is the current positive-step count since the last
    non-positive power event, i.e. ``ntimestep - last_negative`` in
    LAMMPS and ``new_n_pos`` in this implementation.

    .. note::

        ``max_step`` semantics differ from :func:`scale_by_fire`. They
        are **not** drop-in equivalent — a YAML swap between the two
        will silently change the bound applied to position updates:

        * :func:`scale_by_fire` (original): clips the position update
          by **global L2 norm**, ``‖Δx‖₂ ≤ max_step``, applied as a
          single rescale of the full update vector. Default ``0.2`` Å.
        * :func:`scale_by_fire2`, ``use_abc=False``: LAMMPS-style
          ``dmax`` — one-shot timestep rescale based on the
          **∞-norm** of velocity (``max_i |v_i|``), so
          ``max_i |Δx_i| ≤ max_step`` per step. Default ``0.1`` Å.
        * :func:`scale_by_fire2`, ``use_abc=True``: per-component
          **velocity clip** ``|v_i| ≤ max_step / dtv`` that persists
          into the next step. Default ``0.1`` Å.

        Because the ∞-norm bound is per-component while the L2 bound is
        global, an identical ``max_step`` value is *more* restrictive
        under ``scale_by_fire2`` for high-dimensional systems. Tune
        ``max_step`` after switching algorithms.

    Args:
        dt_start: Initial timestep.
        dt_max: Maximum timestep.
        dt_min: Minimum timestep.
        max_step: Maximum displacement bound ``dmax`` (Å). Behaviour
            matches LAMMPS ``MinFire``:

            * ``use_abc=False`` (FIRE 2.0): one-shot timestep rescale —
              compute ``vmax = max_i|v_i|`` (∞-norm).  If
              ``dt · vmax > dmax`` then ``dtv = dmax / vmax``, otherwise
              ``dtv = dt``.  ``dtv`` is used for both the Euler kick and
              the position step (matching LAMMPS).
            * ``use_abc=True`` (ABC-FIRE): per-component velocity clip
              ``v_i ← clamp(v_i, ±dmax/dt)``; the clipped velocity is
              persisted into the next step.

            Set to ``None`` to disable.
        f_inc: Factor to increase ``dt`` when making progress (LAMMPS
            ``dtgrow``).
        f_dec: Factor to decrease ``dt`` on a bad step (LAMMPS
            ``dtshrink``).  LAMMPS only shrinks when
            ``dt * f_dec >= dt_min``; otherwise ``dt`` is left unchanged.
        alpha_start: Initial velocity mixing parameter (LAMMPS
            ``alpha0``).
        f_alpha: Factor to decay alpha when making progress (LAMMPS
            ``alphashrink``).
        n_min: Minimum positive-power steps before increasing ``dt``
            (LAMMPS ``delaystep``).
        use_abc: If True, apply the ABC-FIRE bias correction to the
            mixing step.
        halfstepback: If True (default), apply the FIRE 2.0 half-step
            backtrack ``x -= 0.5·dt·v_old`` on non-positive power steps
            (LAMMPS ``halfstepback yes``).
        delaystep_start: If True (default), suppress ``dt`` shrinking
            and ``alpha`` reset while ``n_total < n_min``
            (LAMMPS ``delaystep_start_flag`` and ``< delaystep``).
            This prevents penalising the mandatory P = 0 first step
            when velocity is zero.

    Returns:
        Optax GradientTransformation implementing FIRE 2.0.

    References:
        * Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020).
        * Echeverri Restrepo & Andric, Comput. Mater. Sci. 218, 111978 (2023).
        * LAMMPS ``src/min_fire.cpp`` (develop branch).
    """

    def init_fn(params: optax.Params) -> ScaleByFire2State:
        return ScaleByFire2State(
            velocity=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt_start),
            alpha=jnp.array(alpha_start),
            n_pos=jnp.array(0, dtype=jnp.int32),
            n_total=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates,
        state: ScaleByFire2State,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByFire2State]:
        del params

        # F = -gradient (forces point downhill)
        forces = jax.tree.map(lambda g: -g, updates)

        n_total = state.n_total + 1

        # ----- P = v_old · F (LAMMPS: vdotfall) -------------------------
        # On iteration 1, v_old = 0 → P = 0 → negative branch.
        power = optax.tree_utils.tree_vdot(forces, state.velocity)
        positive_power = power > 0.0  # type: ignore

        # ----- n_pos (LAMMPS: ntimestep - last_negative) -----------------
        new_n_pos = jnp.where(positive_power, state.n_pos + 1, 0)
        should_increase = jnp.logical_and(positive_power, new_n_pos > n_min)

        # ----- dt adaptation ---------------------------------------------
        # LAMMPS: dt = min(dt*dtgrow, dtmax) when past delay.
        # LAMMPS: if (dt*dtshrink >= dtmin) dt *= dtshrink  (else unchanged)
        dt_increased = jnp.minimum(state.dt * f_inc, dt_max)
        dt_decreased = jnp.where(
            state.dt * f_dec >= dt_min,
            state.dt * f_dec,
            state.dt,  # leave unchanged (LAMMPS behaviour)
        )
        new_dt = jnp.where(
            positive_power,
            jnp.where(should_increase, dt_increased, state.dt),
            dt_decreased,
        )

        # ----- alpha adaptation ------------------------------------------
        # LAMMPS floors alpha before ABC mixing and before persisting the
        # shrunk value for the next iteration.
        alpha_for_mixing = jnp.maximum(state.alpha, 1e-10) if use_abc else state.alpha
        new_alpha = jnp.where(
            positive_power,
            jnp.where(
                should_increase,
                alpha_for_mixing * f_alpha,
                alpha_for_mixing,
            ),
            jnp.array(alpha_start, dtype=state.alpha.dtype),
        )

        # ----- delaystep_start: suppress shrink during startup -----------
        # LAMMPS: if (ntimestep - ntimestep_start < delaystep &&
        #             delaystep_start_flag) delayflag = 0;
        if delaystep_start:
            in_startup = jnp.logical_and(~positive_power, n_total < n_min)
            new_dt = jnp.where(in_startup, state.dt, new_dt)
            new_alpha = jnp.where(in_startup, state.alpha, new_alpha)

        # ----- Mixing scales (LAMMPS: computed before integration) -------
        # LAMMPS: vdotvall, fdotfall use v_old (before Euler kick).
        v_old_sq = optax.tree_utils.tree_vdot(state.velocity, state.velocity)
        f_sq = optax.tree_utils.tree_vdot(forces, forces)

        if use_abc:
            # LAMMPS computes scale1/scale2 only for ``vdotfall > 0`` and
            # using the CURRENT alpha (before shrinking). The shrunk alpha
            # is only persisted for the next iteration.
            abc = jnp.where(
                positive_power,
                1.0 - jnp.power(1.0 - alpha_for_mixing, new_n_pos.astype(new_dt.dtype)),
                1.0,
            )
            safe_abc = jnp.maximum(abc, 1e-30)
            scale1 = jnp.where(positive_power, (1.0 - alpha_for_mixing) / safe_abc, 1.0)
            # LAMMPS: if (fdotfall <= 1e-20) scale2 = 0.0
            scale2_raw = jnp.where(
                f_sq <= 1e-20,  # type: ignore[operator]
                0.0,
                (
                    alpha_for_mixing
                    * jnp.sqrt(v_old_sq / jnp.maximum(f_sq, 1e-20))
                )
                / safe_abc,
            )
            scale2 = jnp.where(positive_power, scale2_raw, 0.0)
        else:
            # LAMMPS: scale1/scale2 use current alpha, not the updated one.
            scale1 = 1.0 - state.alpha
            # LAMMPS: if (fdotfall <= 1e-20) scale2 = 0.0
            scale2 = jnp.where(
                f_sq <= 1e-20,  # type: ignore[operator]
                0.0,
                state.alpha * jnp.sqrt(v_old_sq / jnp.maximum(f_sq, 1e-20)),
            )

        # ----- dmax: compute dtv (LAMMPS: dtvone → dtv) ------------------
        # Non-ABC only.  For ABC dtv = new_dt (no pre-limiting; clip
        # is applied per-component after mixing).
        # LAMMPS checks: for P>0 uses v_old (from previous step);
        # for P<=0 estimates v = dt*F (flagv0 mechanism).
        if max_step is not None and not use_abc:
            vmax_pos = optax.tree_utils.tree_max(
                jax.tree.map(jnp.abs, state.velocity)
            )
            vmax_neg = new_dt * optax.tree_utils.tree_max(
                jax.tree.map(jnp.abs, forces)
            )
            vmax = jnp.where(positive_power, vmax_pos, vmax_neg)
            dtv = jnp.where(
                new_dt * vmax > max_step,
                max_step / jnp.maximum(vmax, 1e-30),
                new_dt,
            )
        else:
            dtv = new_dt

        # ----- Half-step backtrack (P <= 0) ------------------------------
        # LAMMPS: x -= 0.5 * dt * v (dt is already the new/shrunken dt).
        if halfstepback:
            backtrack = jax.tree.map(lambda v: -0.5 * new_dt * v, state.velocity)
        else:
            backtrack = jax.tree.map(jnp.zeros_like, state.velocity)

        # ----- Velocity: zero on P <= 0, keep on P > 0 ------------------
        v_pre = jax.tree.map(
            lambda v: jnp.where(positive_power, v, jnp.zeros_like(v)),
            state.velocity,
        )

        # ----- Euler-implicit kick: v += dtv * F -------------------------
        # LAMMPS: dtf = dtv * ftm2v; v += dtf/mass * f
        # (unit mass, ftm2v = 1 → v += dtv * f)
        v_int = jax.tree.map(lambda v, f: v + dtv * f, v_pre, forces)

        # ----- Mixing (applied only when P > 0, LAMMPS: if vdotfall>0) ---
        v_mixed = jax.tree.map(
            lambda v, f: scale1 * v + scale2 * f, v_int, forces
        )
        new_velocity = jax.tree.map(
            lambda mix, integ: jnp.where(positive_power, mix, integ),
            v_mixed,
            v_int,
        )

        # ----- ABC per-component dmax clip (P > 0 only) ------------------
        # LAMMPS: if (fabs(v[i]*dtv) > dmax) v[i] = dmax/dtv * sign(v[i])
        if max_step is not None and use_abc:
            limit = max_step / jnp.maximum(dtv, 1e-30)
            v_clipped = jax.tree.map(
                lambda v: jnp.clip(v, -limit, limit), new_velocity
            )
            # Clip only on positive-power steps (LAMMPS gates on vdotfall>0)
            new_velocity = jax.tree.map(
                lambda cl, uncl: jnp.where(positive_power, cl, uncl),
                v_clipped,
                new_velocity,
            )

        # ----- Position update: x += dtv * v (+ backtrack on P <= 0) -----
        position_updates = jax.tree.map(
            lambda v, back: dtv * v + jnp.where(
                positive_power, jnp.zeros_like(back), back
            ),
            new_velocity,
            backtrack,
        )

        return position_updates, ScaleByFire2State(
            velocity=new_velocity,
            dt=new_dt,
            alpha=new_alpha,
            n_pos=new_n_pos,
            n_total=n_total,
        )

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore[arg-type]
