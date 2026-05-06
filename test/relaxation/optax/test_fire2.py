# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for FIRE 2.0 / ABC-FIRE optimizer."""

import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.relaxation.optax.fire import ScaleByFire2State, scale_by_fire2

from ...clear_cache import clear_cache  # noqa: F401


class TestScaleByFire2:
    """Tests for scale_by_fire2 transform (FIRE 2.0 / ABC-FIRE)."""

    def test_init(self):
        """init_fn creates correct state, including n_total."""
        optimizer = scale_by_fire2(dt_start=0.1, alpha_start=0.25)
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)
        assert isinstance(state, ScaleByFire2State)
        npt.assert_array_equal(state.velocity, jnp.zeros(3))
        npt.assert_allclose(state.dt, 0.1)
        npt.assert_allclose(state.alpha, 0.25)
        assert state.n_pos == 0
        assert state.n_total == 0

        # PyTree params
        params_tree = {"a": jnp.zeros((4, 3)), "b": jnp.zeros((1, 3, 3))}
        state_tree = optimizer.init(params_tree)
        assert isinstance(state_tree, ScaleByFire2State)
        assert isinstance(state_tree.velocity, dict)
        npt.assert_array_equal(state_tree.velocity["a"], jnp.zeros((4, 3)))  # type: ignore[arg-type]
        npt.assert_array_equal(state_tree.velocity["b"], jnp.zeros((1, 3, 3)))  # type: ignore[arg-type]

    def test_n_total_increments_each_step(self):
        """n_total should increment regardless of branch."""
        optimizer = scale_by_fire2(dt_start=0.1, n_min=2)
        params = jnp.array([1.0])
        state = optimizer.init(params)
        assert isinstance(state, ScaleByFire2State)
        gradient = jnp.array([-1.0])
        for i in range(4):
            _, state = optimizer.update(gradient, state, params)
            assert isinstance(state, ScaleByFire2State)
            assert int(state.n_total) == i + 1

    def test_positive_power_increases_n_pos(self):
        """Consecutive P>0 steps should accumulate n_pos."""
        # Disable delaystep_start so we leave the startup mask quickly.
        optimizer = scale_by_fire2(
            dt_start=0.1, n_min=2, delaystep_start=False
        )
        params = jnp.array([1.0])
        state = optimizer.init(params)
        gradient = jnp.array([-1.0])
        for _ in range(4):
            _, state = optimizer.update(gradient, state, params)
        assert isinstance(state, ScaleByFire2State)
        assert int(state.n_pos) > 0

    def test_dt_increases_after_n_min_positive_steps(self):
        """In P>0 branch, dt grows by f_inc once n_pos > n_min."""
        optimizer = scale_by_fire2(
            dt_start=0.1,
            dt_max=10.0,
            n_min=2,
            f_inc=1.5,
            delaystep_start=False,
            max_step=None,
        )
        params = jnp.array([1.0])
        state = optimizer.init(params)
        assert isinstance(state, ScaleByFire2State)
        initial_dt = float(state.dt)
        gradient = jnp.array([-1.0])
        for _ in range(6):
            _, state = optimizer.update(gradient, state, params)
        assert isinstance(state, ScaleByFire2State)
        assert float(state.dt) > initial_dt

    def test_dt_decreases_on_negative_power(self):
        """In P<=0 branch, dt shrinks by f_dec (after startup mask)."""
        # Manually craft a state past startup with non-zero velocity
        # opposing the force, so power < 0.
        optimizer = scale_by_fire2(
            dt_start=0.1, dt_min=1e-6, f_dec=0.5, n_min=2
        )
        params = jnp.array([1.0])
        state = ScaleByFire2State(
            velocity=jnp.array([1.0]),
            dt=jnp.array(0.1),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(5, dtype=jnp.int32),
            n_total=jnp.array(10, dtype=jnp.int32),
        )
        gradient = jnp.array([1.0])  # F = -1, v=+1 → P = -1 < 0
        _, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        assert float(new_state.dt) < 0.1
        npt.assert_allclose(float(new_state.dt), 0.05)

    def test_dt_bounded_by_dt_min(self):
        """dt should never drop below dt_min (LAMMPS behaviour)."""
        optimizer = scale_by_fire2(
            dt_start=0.02, dt_min=0.01, f_dec=0.5, n_min=1
        )
        params = jnp.array([1.0])
        # Repeatedly force the negative branch.
        state = ScaleByFire2State(
            velocity=jnp.array([1.0]),
            dt=jnp.array(0.02),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(0, dtype=jnp.int32),
            n_total=jnp.array(10, dtype=jnp.int32),
        )
        gradient = jnp.array([1.0])
        for _ in range(10):
            _, state = optimizer.update(gradient, state, params)
            assert isinstance(state, ScaleByFire2State)
            # Re-inject opposing velocity to keep P<=0.
            state = ScaleByFire2State(
                velocity=jnp.array([1.0]),
                dt=state.dt,
                alpha=state.alpha,
                n_pos=state.n_pos,
                n_total=state.n_total,
            )
        assert float(state.dt) >= 0.01 - 1e-6

    def test_halfstepback_applies_on_negative_power(self):
        """With halfstepback=True, P<=0 step contributes -0.5*dt*v_old."""
        optimizer = scale_by_fire2(
            dt_start=0.1,
            max_step=None,
            halfstepback=True,
            delaystep_start=False,
        )
        params = jnp.array([0.0])
        v_old = jnp.array([2.0])
        state = ScaleByFire2State(
            velocity=v_old,
            dt=jnp.array(0.1),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(0, dtype=jnp.int32),
            n_total=jnp.array(10, dtype=jnp.int32),
        )
        gradient = jnp.array([1.0])  # F = -1, v=+2 → P = -2
        updates, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        assert isinstance(updates, jnp.ndarray)

        # On P<=0: v_pre=0, v_int = dtv*F. With max_step=None, dtv=new_dt.
        new_dt = float(new_state.dt)
        v_int = new_dt * (-1.0)
        backtrack = -0.5 * new_dt * float(v_old[0])
        expected = new_dt * v_int + backtrack
        npt.assert_allclose(float(updates[0]), expected, rtol=1e-6)

    def test_halfstepback_disabled(self):
        """halfstepback=False removes the -0.5*dt*v_old contribution."""
        v_old = jnp.array([2.0])
        params = jnp.array([0.0])
        gradient = jnp.array([1.0])

        def _state():
            return ScaleByFire2State(
                velocity=v_old,
                dt=jnp.array(0.1),
                alpha=jnp.array(0.25),
                n_pos=jnp.array(0, dtype=jnp.int32),
                n_total=jnp.array(10, dtype=jnp.int32),
            )

        u_with, s_with = scale_by_fire2(
            dt_start=0.1,
            max_step=None,
            delaystep_start=False,
            halfstepback=True,
        ).update(gradient, _state(), params)
        assert isinstance(s_with, ScaleByFire2State)
        assert isinstance(u_with, jnp.ndarray)
        u_without, _ = scale_by_fire2(
            dt_start=0.1,
            max_step=None,
            delaystep_start=False,
            halfstepback=False,
        ).update(gradient, _state(), params)
        assert isinstance(u_without, jnp.ndarray)
        # The two should differ exactly by the backtrack term, which
        # uses the (possibly shrunk) new_dt, not the original dt.
        diff = float(u_without[0] - u_with[0])
        expected = 0.5 * float(s_with.dt) * float(v_old[0])
        npt.assert_allclose(diff, expected, rtol=1e-6)

    def test_delaystep_start_suppresses_shrink(self):
        """First step has v=0 → P=0 (negative branch) but startup mask
        should keep dt and alpha unchanged when delaystep_start=True."""
        optimizer = scale_by_fire2(
            dt_start=0.1,
            alpha_start=0.25,
            f_dec=0.5,
            n_min=5,
            delaystep_start=True,
        )
        params = jnp.array([1.0])
        state = optimizer.init(params)
        gradient = jnp.array([-1.0])
        _, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        npt.assert_allclose(float(new_state.dt), 0.1)
        npt.assert_allclose(float(new_state.alpha), 0.25)

    def test_delaystep_start_disabled_shrinks_immediately(self):
        """With delaystep_start=False, the very first (P=0) step shrinks dt."""
        optimizer = scale_by_fire2(
            dt_start=0.1,
            f_dec=0.5,
            dt_min=1e-6,
            n_min=5,
            delaystep_start=False,
        )
        params = jnp.array([1.0])
        state = optimizer.init(params)
        gradient = jnp.array([-1.0])
        _, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        npt.assert_allclose(float(new_state.dt), 0.05)

    def test_negative_power_resets_velocity_and_alpha(self):
        """P<=0 zeros velocity (before Euler kick) and resets alpha."""
        alpha_start = 0.25
        optimizer = scale_by_fire2(
            dt_start=0.1,
            alpha_start=alpha_start,
            max_step=None,
            halfstepback=False,
            delaystep_start=False,
        )
        params = jnp.array([1.0])
        state = ScaleByFire2State(
            velocity=jnp.array([5.0]),
            dt=jnp.array(0.1),
            alpha=jnp.array(0.01),  # decayed
            n_pos=jnp.array(7, dtype=jnp.int32),
            n_total=jnp.array(20, dtype=jnp.int32),
        )
        gradient = jnp.array([1.0])  # P = -5 < 0
        _, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        assert int(new_state.n_pos) == 0
        npt.assert_allclose(float(new_state.alpha), alpha_start)
        # v_pre=0, v_int = dtv*F; new_velocity is v_int (no mixing on P<=0).
        v_new = jnp.asarray(new_state.velocity)
        npt.assert_allclose(
            float(v_new[0]),
            float(new_state.dt) * (-1.0),
            rtol=1e-6,
        )

    def test_abc_differs_from_non_abc_at_small_n(self):
        """ABC bias correction should change the mixing at small N.

        At N=1, the ABC scale1 = (1-α)/(1-(1-α)^1) = (1-α)/α, which
        differs strongly from the non-ABC scale1 = 1-α.
        """
        params = jnp.array([1.0, 0.0])
        v_old = jnp.array([1.0, 0.5])
        state_tmpl = ScaleByFire2State(
            velocity=v_old,
            dt=jnp.array(0.1),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(0, dtype=jnp.int32),
            n_total=jnp.array(10, dtype=jnp.int32),
        )
        # Gradient chosen so that F·v > 0 (positive-power branch).
        gradient = jnp.array([-1.0, -0.5])

        opt_plain = scale_by_fire2(
            dt_start=0.1,
            alpha_start=0.25,
            max_step=None,
            delaystep_start=False,
            use_abc=False,
        )
        opt_abc = scale_by_fire2(
            dt_start=0.1,
            alpha_start=0.25,
            max_step=None,
            delaystep_start=False,
            use_abc=True,
        )
        _, s_plain = opt_plain.update(gradient, state_tmpl, params)
        _, s_abc = opt_abc.update(gradient, state_tmpl, params)
        assert isinstance(s_plain, ScaleByFire2State)
        assert isinstance(s_abc, ScaleByFire2State)

        # Both took the positive branch.
        assert int(s_plain.n_pos) == 1
        assert int(s_abc.n_pos) == 1
        # Velocities must differ (bias correction kicks in at N=1).
        v_plain = jnp.asarray(s_plain.velocity)
        v_abc = jnp.asarray(s_abc.velocity)
        diff = float(jnp.linalg.norm(v_abc - v_plain))
        assert diff > 1e-3

    def test_abc_per_component_clip(self):
        """With use_abc=True, max_step clamps |v_i| <= max_step/dtv."""
        max_step = 0.05
        optimizer = scale_by_fire2(
            dt_start=0.1,
            alpha_start=0.25,
            max_step=max_step,
            use_abc=True,
            delaystep_start=False,
        )
        params = jnp.array([0.0, 0.0])
        # Large velocity along force direction → P>0, clip should bite.
        state = ScaleByFire2State(
            velocity=jnp.array([100.0, 0.0]),
            dt=jnp.array(0.1),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(5, dtype=jnp.int32),
            n_total=jnp.array(20, dtype=jnp.int32),
        )
        gradient = jnp.array([-1.0, 0.0])
        _, new_state = optimizer.update(gradient, state, params)
        assert isinstance(new_state, ScaleByFire2State)
        limit = max_step / float(new_state.dt)
        assert (
            float(jnp.max(jnp.abs(jnp.asarray(new_state.velocity))))
            <= limit + 1e-6
        )

    def test_max_step_none_disables_clipping(self):
        """max_step=None should let updates exceed any fixed bound."""
        optimizer = scale_by_fire2(
            dt_start=1.0,
            dt_max=1.0,
            max_step=None,
            delaystep_start=False,
        )
        params = jnp.array([0.0])
        state = ScaleByFire2State(
            velocity=jnp.array([100.0]),
            dt=jnp.array(1.0),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(10, dtype=jnp.int32),
            n_total=jnp.array(20, dtype=jnp.int32),
        )
        gradient = jnp.array([-10.0])  # P > 0
        updates, _ = optimizer.update(gradient, state, params)
        updates_arr = jnp.asarray(updates)
        assert float(jnp.abs(updates_arr[0])) > 1.0

    def test_max_step_clips_non_abc(self):
        """In non-ABC mode, dmax limits dtv such that ||Δx||_∞ ≤ max_step."""
        max_step = 0.1
        optimizer = scale_by_fire2(
            dt_start=1.0,
            dt_max=10.0,
            max_step=max_step,
            use_abc=False,
            delaystep_start=False,
        )
        params = jnp.array([0.0, 0.0])
        state = ScaleByFire2State(
            velocity=jnp.array([100.0, 0.0]),
            dt=jnp.array(1.0),
            alpha=jnp.array(0.25),
            n_pos=jnp.array(10, dtype=jnp.int32),
            n_total=jnp.array(20, dtype=jnp.int32),
        )
        gradient = jnp.array([-1.0, 0.0])
        updates, _ = optimizer.update(gradient, state, params)
        assert float(jnp.max(jnp.abs(jnp.asarray(updates)))) <= max_step + 1e-6

    def test_convergence_on_quadratic(self):
        """FIRE 2.0 should converge on a simple quadratic potential."""
        optimizer = scale_by_fire2(
            dt_start=0.05, dt_max=0.5, max_step=0.5
        )
        x = jnp.array([5.0])
        state = optimizer.init(x)
        for _ in range(200):
            gradient = x
            updates, state = optimizer.update(gradient, state, x)
            x = optax.apply_updates(x, updates)
        npt.assert_allclose(jnp.asarray(x), jnp.zeros(1), atol=1e-2)

    def test_convergence_on_quadratic_abc(self):
        """ABC-FIRE should also converge on a simple quadratic."""
        optimizer = scale_by_fire2(
            dt_start=0.05, dt_max=0.5, max_step=0.5, use_abc=True
        )
        x = jnp.array([5.0])
        state = optimizer.init(x)
        for _ in range(200):
            gradient = x
            updates, state = optimizer.update(gradient, state, x)
            x = optax.apply_updates(x, updates)
        npt.assert_allclose(jnp.asarray(x), jnp.zeros(1), atol=1e-2)
