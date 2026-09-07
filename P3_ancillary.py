#!/usr/bin/env python
"""
Wednesday's job is to point an optimiser at Monday's surrogate and find out
what happens. This module is the plumbing between the two. You will edit
exactly two things: the bounds you give the parameterizer, and the objective.
Everything else is here so you do not spend the afternoon writing gym
boilerplate.

    P1_ancillary   geometry, the surrogate contract, the envelope. Imported,
                   never duplicated -- if you broke it on Monday, re-download
                   it, because P3 imports the same file.
    P3_physics     the referee. It is a MOCK high-fidelity model; read the
                   header of that file before you believe anything it says
                   about aerodynamics.

WHAT CONNECTS TO WHAT
---------------------
    NACA4Parameterizer   turns 3 numbers (m, p, t) into a shape and back.
                         Its bounds are the agent's action space -- rung 1.
    Objective            turns a shape into one number. This is the reward,
                         AND it is what the DE baseline minimises. One object,
                         both consumers, which is the only way the comparison
                         means anything -- rungs 2, 3 and 4.
    SurrogateSolver      the adapter pyLOM's environment expects.

pyLOM's `ShapeOptimizationEnv` computes reward as the *improvement*
solver(shape_t) - solver(shape_t-1), so an episode's return is just
final - initial. Report `objective(final)` as the score, never the return, or
you are scoring where the agent started.

Leave `thickness_penalization_factor` at 0. pyLOM applies it outside the
solver, so DE would never see it and the two optimisers would silently be
solving different problems. Every constraint you want belongs in the Objective.

A NOTE ON PARALLELISM
---------------------
pyLOM's `create_env` uses SubprocVecEnv above one worker, which needs a
`__main__` guard and a picklable surrogate -- in a notebook it hangs. Use
`make_vec_env` below, which is DummyVecEnv. Your surrogate is a small MLP;
inference is microseconds and the bottleneck is Python, so separate processes
buy you almost nothing here anyway.
"""

from __future__ import annotations

import numpy as np

import P1_ancillary as bip
import P3_physics as phys

__all__ = [
    "NACA4Shape", "NACA4Parameterizer",
    "Objective", "LiftToDragObjective", "CallCounter",
    "RefereeCorrectedSurrogate", "SurrogateSolver",
    "make_env", "make_vec_env", "make_contextual_env", "rollout",
    "set_seeds", "mf_loop",
    "SurrogateProblem", "run_de", "comparison_table",
    "load_surrogate", "check_p3_contract", "animate_evolution",
    "ALPHA_DESIGN", "NON_CONVERGED_REWARD", "PPO_KWARGS", "TRAIN_STEPS",
]

#: The angle of attack runs 1 and 2 are flown at. Inside the P1 clamp.
ALPHA_DESIGN = 5.0

#: Matches pyLOM's sentinel. A solver returning this makes the environment
#: revert the step and truncate the episode. Reserved for geometric nonsense.
NON_CONVERGED_REWARD = -100000


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
class NACA4Shape:
    """A NACA 4-digit section that quacks enough like an `aerosandbox.Airfoil`.

    pyLOM's environment hands a *shape* to the solver, but your surrogate wants
    *numbers*, and pyLOM's own progress plot wants `.x()` and `.y()`. This
    object carries both so nothing downstream has to be rewritten.

    `params` is (m [%], p [-], t [%]) in P1 units throughout. Angle of attack
    is not a shape property and lives on the solver.
    """

    __slots__ = ("params", "_coords")

    def __init__(self, params: np.ndarray, n_points: int = 161):
        self.params = np.asarray(params, dtype=np.float64).ravel()[:3]
        m, p, t = self.params
        self._coords = bip.naca4_coordinates(m, p, t, n_points=n_points)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coords

    def x(self) -> np.ndarray:
        return self._coords[:, 0]

    def y(self) -> np.ndarray:
        return self._coords[:, 1]

    def max_thickness(self) -> float:
        """Fraction of chord, matching aerosandbox's convention (0.12, not 12)."""
        return float(self.params[2] / 100.0)

    def name(self) -> str:
        return bip.naca4_name(*self.params)

    def query(self, alpha: float) -> np.ndarray:
        """(1, 4) row ready for a surrogate or the referee."""
        return np.array([[*self.params, float(alpha)]], dtype=np.float64)

    def is_valid(self) -> bool:
        """Cheap geometric sanity. Not a performance judgement.

        Reserved for shapes that are not airfoils at all -- zero thickness,
        non-finite coordinates, no enclosed area. A shape that is merely *bad*
        must stay valid, because the agent finding bad shapes attractive is the
        thing we are trying to observe.

        Note there is no self-intersection test: a NACA 4-digit section cannot
        cross itself, because thickness is applied normal to the camber line
        from a strictly positive distribution. Upper and lower points do not
        even share x-stations on a cambered section, so the obvious pointwise
        comparison is wrong as well as unnecessary.
        """
        m, p, t = self.params
        if not np.all(np.isfinite(self.params)) or t <= 0.5:
            return False
        if not np.isfinite(self._coords).all():
            return False
        x, y = self._coords[:, 0], self._coords[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return bool(area > 1e-4)

    def __repr__(self) -> str:
        return f"<NACA4Shape {self.name()}>"


class NACA4Parameterizer:
    """(m, p, t) <-> shape, with the bounds that define the action space.

    Duck-typed against pyLOM's `BaseParameterizer` -- the environment never
    isinstance-checks, so this needs no pyLOM import and stays testable on its
    own.

    THE BOUNDS ARE RUNG 1. Handing the agent the full P1 clamp lets it reach
    geometries your campaign may never have sampled; narrowing them to the box
    your own data actually covers is the cheapest fix on the ladder, and it
    costs one constructor argument.

    Everything is float32 on the way out. pyLOM's `step` clips to float32 while
    `reset` returns whatever this produces, and gymnasium's passive checker
    complains about the mismatch at the least convenient moment.
    """

    KEYS = ("m", "p", "t")

    def __init__(self, bounds: dict | None = None, n_points: int = 161):
        b = bounds or bip.CLAMP
        self.bounds = {k: (float(b[k][0]), float(b[k][1])) for k in self.KEYS}
        self.n_points = int(n_points)
        self._lo = np.array([self.bounds[k][0] for k in self.KEYS], dtype=np.float32)
        self._hi = np.array([self.bounds[k][1] for k in self.KEYS], dtype=np.float32)

    def get_optimizable_bounds(self):
        return [self._lo.tolist(), self._hi.tolist()]

    def get_shape_from_params(self, params) -> NACA4Shape:
        return NACA4Shape(np.clip(np.asarray(params, dtype=np.float64),
                                  self._lo, self._hi), self.n_points)

    def get_params_from_shape(self, shape) -> np.ndarray:
        if isinstance(shape, NACA4Shape):
            p = shape.params
        else:                                   # a bare (m, p, t) triple
            p = np.asarray(shape, dtype=np.float64).ravel()[:3]
        return np.clip(p, self._lo, self._hi).astype(np.float32)

    def generate_random_params(self, seed=None) -> np.ndarray:
        """Uniform in the box. Random starts, not a fixed NACA 0012.

        A policy that only ever starts from one section learns a trajectory,
        not a policy. Evaluate from a fixed start instead, via
        `env.reset(options={"initial_shape": ...})`.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self._lo, self._hi).astype(np.float32)


# --------------------------------------------------------------------------
# the objective -- one object, shared by the agent and the baseline
# --------------------------------------------------------------------------
class CallCounter:
    """Counts surrogate evaluations. Half of the RL-versus-DE argument.

    DE finds a good section in roughly 800 evaluations. PPO needs tens of
    thousands. Both numbers belong in the table; a comparison that reports only
    the winner is not a comparison.
    """

    def __init__(self, surrogate):
        self.surrogate = surrogate
        self.n_predict = 0
        self.n_rows = 0

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.n_predict += 1
        self.n_rows += X.shape[0]
        return self.surrogate.predict(X)

    def in_envelope(self, X):
        return self.surrogate.in_envelope(X)

    def envelope_score(self, X):
        return self.surrogate.envelope_score(X)

    def lift_to_drag(self, X):
        out = self.predict(X)
        return out[:, 0] / np.maximum(10.0 ** out[:, 1], bip.EPS)

    def reset_counts(self):
        self.n_predict = self.n_rows = 0


class Objective:
    """(N, 4) -> (N,), higher is better. Subclass or configure; do not fork.

    Both the RL environment and the DE baseline call the *same instance*. If
    you find yourself writing a second copy of the reward for the optimiser,
    stop: the comparison stops meaning anything the moment the two differ, and
    `fairness_audit` in the instructor module exists to catch exactly that.

    Vectorised because pymoo evaluates a whole population at once. The RL
    solver calls it with a single row.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def terms(self, X: np.ndarray) -> dict:
        """Per-term breakdown, for working out *why* the agent did that."""
        raise NotImplementedError


class LiftToDragObjective(Objective):
    """Maximise Cl/Cd at a fixed angle, with the fix ladder as arguments.

    The ladder, in the order you will try it:

        rung 1  action bounds        -> NACA4Parameterizer(bounds=...), not here
        rung 2  design requirements  -> `min_t`, `cl_min`
        rung 3  envelope penalty     -> `w_envelope`
        rung 4  referee in the loop  -> wrap the surrogate in
                                        RefereeCorrectedSurrogate

    Rung 3 uses `envelope_score`, not `in_envelope`, so the penalty has a slope.
    A boolean gives an agent one step outside the envelope the same punishment
    as an agent halfway to nowhere, and nothing to follow back in.

    Be warned about rung 3 on this problem: it will not save you. The
    interesting failure on Wednesday lives *inside* the training data, where
    the envelope is perfectly happy. Run it anyway -- finding out what a fix
    does not fix is worth more than another fix that works.
    """

    def __init__(self, surrogate, alpha: float = ALPHA_DESIGN,
                 min_t: float | None = None, cl_min: float | None = None,
                 w_requirement: float = 200.0,
                 w_envelope: float = 0.0, envelope_margin: float = 1.0):
        self.surrogate = surrogate
        self.alpha = float(alpha)
        self.min_t = min_t
        self.cl_min = cl_min
        self.w_requirement = float(w_requirement)
        self.w_envelope = float(w_envelope)
        self.envelope_margin = float(envelope_margin)

    def _rows(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] == 3:                     # alpha not supplied -> design point
            X = np.column_stack([X, np.full(len(X), self.alpha)])
        return X

    def terms(self, X: np.ndarray) -> dict:
        X = self._rows(X)
        out = np.asarray(self.surrogate.predict(X), dtype=np.float64)
        cl, cd = out[:, 0], np.maximum(10.0 ** out[:, 1], bip.EPS)
        base = cl / cd

        pen_t = np.zeros(len(X))
        if self.min_t is not None:
            pen_t = self.w_requirement * np.maximum(0.0, self.min_t - X[:, 2])

        pen_cl = np.zeros(len(X))
        if self.cl_min is not None:
            pen_cl = self.w_requirement * np.maximum(0.0, self.cl_min - cl)

        pen_env = np.zeros(len(X))
        if self.w_envelope > 0.0:
            s = np.asarray(self.surrogate.envelope_score(X), dtype=np.float64)
            pen_env = self.w_envelope * np.maximum(0.0, s - self.envelope_margin) ** 2

        return {"lift_to_drag": base, "cl": cl, "cd": cd,
                "penalty_thickness": pen_t, "penalty_cl": pen_cl,
                "penalty_envelope": pen_env,
                "total": base - pen_t - pen_cl - pen_env}

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.terms(X)["total"]


class RefereeCorrectedSurrogate(bip.AeroSurrogate):
    """Rung 4. Your surrogate, corrected by a handful of referee evaluations.

    This is the only rung that can fix an error your surrogate inherited from
    the model that trained it, because it is the only one that consults
    something better. Rungs 1 to 3 all reason from data you already had.

    The correction is an additive bridge in the surrogate's own output space --
    `Cl` offset and `log10 Cd` offset at each anchor, blended by a Gaussian
    kernel in normalised design space. Deliberately the simplest thing that
    works: with twenty anchors a co-Kriging model would be fitting noise, and
    you can explain this one at a whiteboard.

    WHERE YOU SPEND THE CALLS IS THE WHOLE LESSON. Measured on the reference
    surrogate, three rounds of 25 calls scattered around the incumbent:

        rung 0, no correction     NACA 6409, delivers 143.7, gap 1.37
        round 1                   t = 12.9,  delivers 181,   gap 0.92
        round 2                   t = 12.0,  delivers 184,   gap 0.99
        round 3                   t = 11.8,  delivers 183,   gap 1.00

    against a best-achievable 183.5. Seventy-five high-fidelity evaluations
    turned a 28% performance loss into nothing.

    Two ways to get this wrong, both worth trying so you see them:

        `length_scale` too large (0.6) smears the correction over the whole
        space, the gradient washes out, and the design never moves off t = 9 --
        the gap closes to 1.04 while still delivering 148. An honest surrogate
        that is honest about a bad design has bought you nothing.

        Spreading round 1 uniformly over the clamp instead of around the
        incumbent wastes it: 25 scattered points do not resolve anything, and
        you spend a third of the budget learning the correction is small
        almost everywhere.

    Honesty and discovery are not the same thing, and only the second one
    changes the wing.
    """

    def __init__(self, surrogate, referee=None, length_scale: float = 0.15,
                 max_calls: int = 75):
        self.surrogate = surrogate
        self.referee = referee or phys.Referee()
        self.length_scale = float(length_scale)
        self.max_calls = int(max_calls)
        self.anchors = np.empty((0, 4))
        self.offsets = np.empty((0, 2))         # [dCl, d log10 Cd]
        self._scale = np.array([6.0, 0.5, 10.0, 10.0])   # clamp spans

    @property
    def n_calls(self) -> int:
        return int(self.anchors.shape[0])

    def refine(self, X: np.ndarray) -> int:
        """Spend referee calls on these rows. Returns how many were spent."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        room = max(0, self.max_calls - self.n_calls)
        if room == 0:
            return 0
        X = X[:room]
        Y, _ = self.referee(X)
        base = np.asarray(self.surrogate.predict(X), dtype=np.float64)
        off = np.column_stack([Y[:, 0] - base[:, 0],
                               np.log10(np.maximum(Y[:, 1], bip.EPS)) - base[:, 1]])
        self.anchors = np.vstack([self.anchors, X])
        self.offsets = np.vstack([self.offsets, off])
        return int(X.shape[0])

    def refine_around(self, x, n: int = 25, spread=(0.3, 0.03, 0.8),
                      alpha: float = ALPHA_DESIGN, bounds: dict | None = None,
                      rng=None) -> int:
        """Scatter `n` referee calls around a design and absorb the answers.

        The spread matters as much as the count. Too tight and every anchor
        says the same thing; too wide and you are sampling the clamp rather
        than the neighbourhood. Defaults are roughly 5% of each clamp span.
        """
        b = bounds or bip.CLAMP
        lo = np.array([b[k][0] for k in ("m", "p", "t")], dtype=np.float64)
        hi = np.array([b[k][1] for k in ("m", "p", "t")], dtype=np.float64)
        rng = np.random.default_rng(rng)
        x = np.asarray(x, dtype=np.float64).ravel()[:3]
        A = np.clip(x + rng.normal(0.0, spread, size=(int(n), 3)), lo, hi)
        return self.refine(np.column_stack([A, np.full(len(A), float(alpha))]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        base = np.asarray(self.surrogate.predict(X), dtype=np.float64).copy()
        if self.n_calls == 0:
            return base
        d = np.linalg.norm((X[:, None, :] - self.anchors[None, :, :])
                           / self._scale, axis=2)
        w = np.exp(-(d / self.length_scale) ** 2)
        tot = w.sum(axis=1, keepdims=True)
        blend = np.where(tot > 1e-12, tot, 1.0)
        return base + (w @ self.offsets) / blend

    def in_envelope(self, X: np.ndarray) -> np.ndarray:
        return self.surrogate.in_envelope(X)

    def envelope_score(self, X: np.ndarray) -> np.ndarray:
        return self.surrogate.envelope_score(X)


class SurrogateSolver:
    """Adapter: pyLOM hands it a shape, the Objective wants a row.

    Returns pyLOM's non-convergence sentinel only for geometric nonsense, which
    makes the environment revert the step and end the episode. It does NOT
    trigger on an implausible *prediction*: if the surrogate promises L/D = 900,
    that promise is the entire lesson, and truncating there would hide it.
    """

    NON_CONVERGED_REWARD = NON_CONVERGED_REWARD

    def __init__(self, objective: Objective, alpha: float = ALPHA_DESIGN):
        self.objective = objective
        self.alpha = float(alpha)

    def __call__(self, shape) -> float:
        if isinstance(shape, NACA4Shape) and not shape.is_valid():
            return float(self.NON_CONVERGED_REWARD)
        params = shape.params if isinstance(shape, NACA4Shape) else shape
        row = np.array([[*np.asarray(params, dtype=np.float64).ravel()[:3],
                         self.alpha]])
        val = float(np.asarray(self.objective(row)).ravel()[0])
        return val if np.isfinite(val) else float(self.NON_CONVERGED_REWARD)


# --------------------------------------------------------------------------
# environment construction
# --------------------------------------------------------------------------
def mf_loop(surrogate, referee=None, bounds: dict | None = None,
            rounds: int = 3, n_per_round: int = 25,
            alpha: float = ALPHA_DESIGN, seed: int = 0,
            optimiser=None, **objective_kwargs) -> dict:
    """Rung 4 end to end: optimise, referee the answer, correct, repeat.

    The outer loop is the point. A single big batch of referee calls placed
    before you know where the optimiser wants to go is worse than the same
    budget spent in three rounds that chase it -- the anchors have to land
    where the design is heading, and after round zero you do not yet know.

    `optimiser` defaults to `run_de`; pass your trained agent's search instead
    to make the comparison like-for-like.
    """
    ref = referee or phys.Referee()
    mf = RefereeCorrectedSurrogate(surrogate, referee=ref,
                                   max_calls=rounds * n_per_round)
    opt = optimiser or (lambda obj: run_de(obj, bounds=bounds, seed=seed))

    x = opt(LiftToDragObjective(surrogate, alpha=alpha, **objective_kwargs))["params"]
    history = []
    for r in range(int(rounds)):
        mf.refine_around(x, n=n_per_round, alpha=alpha, bounds=bounds, rng=seed + r)
        res = opt(LiftToDragObjective(mf, alpha=alpha, **objective_kwargs))
        x = res["params"]
        g = phys.promise_gap(mf, NACA4Shape(x).query(alpha), ref)
        history.append({"round": r + 1, "params": x.copy(),
                        "promised": float(g["promised"][0]),
                        "delivered": float(g["delivered"][0]),
                        "gap": float(g["gap"][0]),
                        "referee_calls": mf.n_calls})
    return {"surrogate": mf, "params": x, "shape": NACA4Shape(x),
            "history": history}


def set_seeds(seed: int) -> None:
    """Seed every global generator pyLOM's environment reaches for.

    Its `seed()` sets numpy's global state and `generate_random_params` touches
    `random`, so per-configuration seeding has to happen out here. Seed the
    SB3 model too -- pass `seed=` to the algorithm -- or your "two seeds" are
    one seed twice.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def _shape_env_class():
    """Resolve pyLOM's `ShapeOptimizationEnv` whichever way it is exported.

    `pyLOM.RL.__init__` does `from . import shape_optimization_env`, so the
    name bound at package level is the *module*, not the class -- the obvious
    `from pyLOM.RL import ShapeOptimizationEnv` fails, and importing the module
    under that alias gets you a `'module' object is not callable` at
    construction time, which is a confusing way to find out.

    The class lives at `pyLOM.RL.shape_optimization_env.ShapeOptimizationEnv`
    and is also registered as the gym id "ShapeOptimizationEnv-v0". This tries
    the package attribute first so a future pyLOM that exports the class
    directly keeps working.
    """
    import pyLOM.RL as rl

    cls = getattr(rl, "ShapeOptimizationEnv", None)
    if isinstance(cls, type):
        return cls
    from pyLOM.RL.shape_optimization_env import ShapeOptimizationEnv
    return ShapeOptimizationEnv


def make_env(objective: Objective, bounds: dict | None = None,
             episode_max_length: int = 64, alpha: float = ALPHA_DESIGN,
             seed=None, monitor: bool = True):
    """One `ShapeOptimizationEnv` wired to your objective.

    Wrapped in SB3's `Monitor` by default so `verbose=1` actually reports
    `ep_rew_mean`. Without it PPO prints timing and nothing about whether the
    agent is learning, and on a problem this easy an undertrained agent still
    reaches the clamp corner -- so the endpoint looks fine while the policy is
    barely trained. Watch the curve, not the final design.
    """
    env = _shape_env_class()(
        solver=SurrogateSolver(objective, alpha=alpha),
        parameterizer=NACA4Parameterizer(bounds),
        episode_max_length=episode_max_length,
        thickness_penalization_factor=0,          # see module header
        seed=seed,
    )
    if monitor:
        try:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        except ImportError:
            pass
    return env


#: PPO settings sized for a two-hour practical, not for a paper.
#:
#: SB3 defaults to `n_steps=2048`, which with 8 environments means a 16,384
#: transition rollout buffer and therefore about SIX policy updates in 100,000
#: timesteps. That is seven minutes of compute for six gradient steps. At
#: `n_steps=256` the buffer is 2048 and 30,000 timesteps buys roughly fifteen
#: updates -- more learning, in a fifth of the time.
#:
#: The lesson is worth stating to the students: on a cheap environment the
#: default hyperparameters of an RL library are tuned for expensive ones, and
#: the wrong knob costs you an order of magnitude.
PPO_KWARGS = dict(n_steps=256, batch_size=256, n_epochs=10,
                  learning_rate=3e-4, gae_lambda=0.95)

#: Enough to learn this problem at PPO_KWARGS. Roughly two minutes.
TRAIN_STEPS = 30_000


def make_vec_env(objective: Objective, n_envs: int = 8, **kwargs):
    """DummyVecEnv, not SubprocVecEnv. See the note in the module header."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([
        (lambda i=i: make_env(objective, seed=None if kwargs.get("seed") is None
                              else kwargs["seed"] + i,
                              **{k: v for k, v in kwargs.items() if k != "seed"}))
        for i in range(n_envs)
    ])


def make_contextual_env(objective: Objective, bounds: dict | None = None,
                        episode_max_length: int = 64,
                        alpha_range=(0.0, 10.0), seed=None):
    """The stretch: one policy that designs for whatever angle it is handed.

    Angle of attack becomes *context*, not an action. It is drawn once per
    episode, appended to the observation, and the agent never gets to change
    it -- so the policy has to learn a family of sections indexed by flight
    condition rather than one good answer.

    That is the whole argument for reinforcement learning over a classical
    optimiser on this problem, and it is the only place today where it holds.
    DE beats the agent at finding a single section and will keep beating it;
    but DE has to be re-run from scratch for every angle, while the policy
    answers a new one in a single forward pass. Compare them on that, not on
    the single-point result.

    The observation becomes (m, p, t, alpha). The action stays three-
    dimensional, because pyLOM sizes the action space from the parameterizer
    and alpha is not a shape parameter.

    Evaluate with `env.reset(options={"alpha": 3.0})` to pin the condition;
    leave it out during training so the agent sees the whole range.
    """
    import gymnasium as gym

    lo_a, hi_a = float(alpha_range[0]), float(alpha_range[1])

    class ContextualShapeOptimizationEnv(_shape_env_class()):
        def __init__(self, **kw):
            super().__init__(**kw)
            lo, hi = self.params_bounds
            self.alpha_range = (lo_a, hi_a)
            self.observation_space = gym.spaces.Box(
                low=np.array([*lo, lo_a], dtype=np.float32),
                high=np.array([*hi, hi_a], dtype=np.float32))
            self._alpha = 0.5 * (lo_a + hi_a)

        def _obs(self, params):
            return np.asarray([*np.asarray(params).ravel()[:3], self._alpha],
                              dtype=np.float32)

        def reset(self, seed=None, options=None):
            if options is not None and "alpha" in options:
                self._alpha = float(options["alpha"])
            else:
                self._alpha = float(np.random.default_rng(seed).uniform(lo_a, hi_a))
            # must land on the solver BEFORE super() evaluates the initial shape
            self.solver.alpha = self._alpha
            obs, info = super().reset(seed=seed, options=options)
            info["alpha"] = self._alpha
            return self._obs(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info["alpha"] = self._alpha
            return self._obs(obs), reward, terminated, truncated, info

    return ContextualShapeOptimizationEnv(
        solver=SurrogateSolver(objective, alpha=0.5 * (lo_a + hi_a)),
        parameterizer=NACA4Parameterizer(bounds),
        episode_max_length=episode_max_length,
        thickness_penalization_factor=0,
        seed=seed,
    )


def rollout(model, env, initial_shape=None, deterministic: bool = True,
            reset_options: dict | None = None) -> dict:
    """One episode. Records the whole trajectory, not just the endpoint.

    Pass `initial_shape` to evaluate from a fixed section -- otherwise the
    episode starts somewhere random and the score you report is partly luck
    about where it began.

    `reset_options` goes straight through to `env.reset`. On the contextual
    environment that is how you pin the flight condition:
    `reset_options={"alpha": 3.0}`. Without it the angle is redrawn at random
    and you are not measuring what you think you are.
    """
    opts = dict(reset_options or {})
    if initial_shape is not None:
        opts["initial_shape"] = initial_shape
    obs, info = env.reset(options=opts or None)
    params = [np.asarray(obs, dtype=np.float64).copy()]
    values = [float(info.get("initial_reward", np.nan))]
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(action)
        params.append(np.asarray(obs, dtype=np.float64).copy())
        rewards.append(float(r))
        values.append(values[-1] + float(r))
        done = bool(terminated or truncated)
    P = np.array(params)
    return {"params": P, "rewards": np.array(rewards),
            "objective": np.array(values), "shapes": [NACA4Shape(p) for p in P],
            "best": P[int(np.nanargmax(values))]}


# --------------------------------------------------------------------------
# the fairness baseline
# --------------------------------------------------------------------------
def _pymoo_problem(objective: Objective, bounds: dict | None):
    from pymoo.core.problem import Problem

    par = NACA4Parameterizer(bounds)
    lo, hi = par.get_optimizable_bounds()

    class SurrogateProblem(Problem):
        """Minimise -objective. Batched: `Problem`, never `ElementwiseProblem`.

        Same objective *instance* the agent is rewarded by. Not a copy, not a
        reimplementation -- the instance.
        """

        def __init__(self):
            super().__init__(n_var=3, n_obj=1, xl=np.array(lo, dtype=np.float64),
                             xu=np.array(hi, dtype=np.float64))
            self.objective = objective

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = -np.asarray(self.objective(X), dtype=np.float64).reshape(-1, 1)

    return SurrogateProblem()


SurrogateProblem = _pymoo_problem      # kept as the documented name


def run_de(objective: Objective, bounds: dict | None = None,
           pop_size: int = 20, n_gen: int = 40, seed: int = 0) -> dict:
    """Differential evolution on the identical objective. The honest baseline.

    Default budget is 800 evaluations, which is what DE needs here and a small
    fraction of what PPO will spend. Do not inflate it to make the race look
    close -- the evaluation count is the interesting part of the result.
    """
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.optimize import minimize

    problem = _pymoo_problem(objective, bounds)
    res = minimize(problem, DE(pop_size=pop_size), ("n_gen", n_gen),
                   seed=seed, verbose=False)
    x = np.asarray(res.X, dtype=np.float64).ravel()[:3]
    return {"params": x, "objective": float(-res.F.ravel()[0]),
            "n_eval": int(pop_size * n_gen), "shape": NACA4Shape(x)}


def comparison_table(entries: dict, alpha: float = ALPHA_DESIGN,
                     referee=None, surrogate=None) -> str:
    """Side-by-side of what each optimiser promised and what it delivered.

    `entries` maps a label to a (m, p, t) triple. Reports promise and delivery
    separately on purpose: a design can be over-promised and still good, and a
    design can be honestly predicted and still poor.
    """
    ref = referee or phys.Referee()
    rows = []
    for name, p in entries.items():
        X = np.array([[*np.asarray(p, dtype=np.float64).ravel()[:3], alpha]])
        delivered = float(ref.lift_to_drag(X)[0])
        promised = (float(np.asarray(surrogate.lift_to_drag(X)).ravel()[0])
                    if surrogate is not None else np.nan)
        rows.append((name, X[0, 0], X[0, 1], X[0, 2], promised, delivered,
                     promised / delivered if surrogate is not None else np.nan,
                     bool(surrogate.in_envelope(X)[0]) if surrogate is not None
                     else True))
    w = max(len(r[0]) for r in rows)
    head = (f"{'design':<{w}}  {'m':>5} {'p':>5} {'t':>5} | "
            f"{'promised':>9} {'delivered':>9} {'gap':>5} | in_env")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r[0]:<{w}}  {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.2f} | "
                   f"{r[4]:9.1f} {r[5]:9.1f} {r[6]:5.2f} | {r[7]}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# odds and ends
# --------------------------------------------------------------------------
def load_surrogate(path: str, fallback: str | None = "P1_data/P3_clean"):
    """Your surrogate if it loads, the shipped reference if it does not.

    Nobody sits out Practical 3 because Monday's training run did not converge.
    Say out loud which one you are using when you report numbers.
    """
    try:
        s = bip.PyLOMSurrogate.load(path)
        if bip.check_contract(s, verbose=False):
            return s, "own"
        print(f"! {path} loaded but fails check_contract; using the reference")
    except Exception as exc:                       # noqa: BLE001
        print(f"! could not load {path} ({type(exc).__name__}); using the reference")
    if fallback is None:
        raise RuntimeError("no surrogate available")
    return bip.PyLOMSurrogate.load(fallback), "reference"


def check_p3_contract(surrogate, verbose: bool = True) -> bool:
    """Everything P3 needs, on top of `check_contract`. Run it before training."""
    ok = bip.check_contract(surrogate, verbose=verbose)
    X = np.array([[3.0, 0.4, 12.0, 5.0], [6.0, 0.4, 8.0, 5.0]])
    try:
        s = np.asarray(surrogate.envelope_score(X), dtype=np.float64)
        assert s.shape == (2,) and np.all(np.isfinite(s)) and np.all(s >= 0)
        if verbose:
            graded = float(s.min()) != float(s.max()) or not np.all(np.isin(s, (0.0, 1.0)))
            print(f"  envelope_score ok, graded: {graded}"
                  + ("" if graded else "  (binary fallback -- rung 4 will be blunt)"))
    except Exception as exc:                       # noqa: BLE001
        print(f"  envelope_score FAILED: {exc}")
        ok = False
    try:
        obj = LiftToDragObjective(surrogate)
        v = obj(np.array([[3.0, 0.4, 12.0]]))
        assert np.isfinite(v).all()
        if verbose:
            print(f"  objective evaluates: L/D = {float(v[0]):.1f}")
    except Exception as exc:                       # noqa: BLE001
        print(f"  objective FAILED: {exc}")
        ok = False
    return ok


def animate_evolution(shapes, values=None, path: str = "evolution.gif",
                      fps: int = 8, title: str | None = None):
    """Trajectory as a GIF. Matplotlib, not manim.

    pyLOM ships `AirfoilEvolutionAnimation`, which is a manim Scene and wants
    LaTeX and ffmpeg -- not something to install during a session. This does
    the same job with the writer everyone already has.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    shapes = list(shapes)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    txt = ax.set_title(title or "")

    def update(i):
        s = shapes[i]
        line.set_data(s.x(), s.y())
        lbl = f"{i:3d}/{len(shapes) - 1}  {s.name()}"
        if values is not None and i < len(values) and np.isfinite(values[i]):
            lbl += f"   objective {values[i]:.1f}"
        txt.set_text(lbl if title is None else f"{title}\n{lbl}")
        return line, txt

    anim = FuncAnimation(fig, update, frames=len(shapes), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _self_test() -> None:
    from P1_instructor import lhs, scale_to

    sh = NACA4Shape([6.0, 0.4, 12.0])
    print(f"{sh}  max_thickness {sh.max_thickness():.3f}  valid {sh.is_valid()}")
    assert abs(sh.max_thickness() - 0.12) < 1e-3
    assert sh.is_valid(), "a NACA 6412 must be a valid section"
    assert not NACA4Shape([3.0, 0.4, 0.2]).is_valid()
    corners = [(m, p, t) for m in (0.0, 6.0) for p in (0.2, 0.7)
               for t in (8.0, 18.0)]
    assert all(NACA4Shape(c).is_valid() for c in corners), \
        "every corner of the clamp must be a valid section"

    par = NACA4Parameterizer()
    lo, hi = par.get_optimizable_bounds()
    print(f"bounds {lo} .. {hi}")
    p = par.generate_random_params(seed=0)
    assert p.dtype == np.float32, "reset dtype must be float32"
    assert np.allclose(par.get_params_from_shape(par.get_shape_from_params(p)),
                       p, atol=1e-5), "params -> shape -> params must round-trip"

    # a stand-in surrogate; the real one is a pyLOM MLP
    class Fake(bip.AeroSurrogate):
        def predict(self, X):
            X = np.atleast_2d(X)
            cl = 0.1 * X[:, 3] + 0.09 * X[:, 0]
            cd = -2.2 + 0.02 * X[:, 2]
            return np.column_stack([cl, cd])

        def in_envelope(self, X):
            X = np.atleast_2d(X)
            return X[:, 2] <= 16.0

    fake = CallCounter(Fake())
    obj = LiftToDragObjective(fake)
    X = scale_to(lhs(64, 3, np.random.default_rng(0)),
                 [bip.CLAMP[k] for k in ("m", "p", "t")])
    v = obj(X)
    print(f"objective on {len(X)} rows: [{v.min():.1f}, {v.max():.1f}]  "
          f"surrogate calls {fake.n_predict}")

    solver = SurrogateSolver(obj)
    assert abs(solver(NACA4Shape(X[0])) - obj(X[:1])[0]) < 1e-9, "solver must agree"
    assert solver(NACA4Shape([3.0, 0.4, 0.2])) == NON_CONVERGED_REWARD

    # the fairness property, checked rather than asserted in prose
    prob = _pymoo_problem(obj, None)
    out = {}
    prob._evaluate(X, out)
    assert np.allclose(-out["F"].ravel(), obj(X)), "pymoo and the agent disagree"
    print("pymoo problem and RL solver share one objective: True")

    de = run_de(obj, pop_size=10, n_gen=10, seed=0)
    print(f"DE -> {de['shape'].name()}  objective {de['objective']:.1f} "
          f"in {de['n_eval']} evals")

    # rungs change behaviour
    strict = LiftToDragObjective(fake, min_t=12.0, w_envelope=50.0)
    t = strict.terms(np.array([[6.0, 0.4, 8.0]]))
    assert t["penalty_thickness"][0] > 0, "rung 2 must bite below min_t"
    print(f"rung 2 penalty at t=8 with min_t=12: {t['penalty_thickness'][0]:.1f}")
    print("check_p3_contract:", check_p3_contract(Fake(), verbose=True))


if __name__ == "__main__":
    _self_test()
