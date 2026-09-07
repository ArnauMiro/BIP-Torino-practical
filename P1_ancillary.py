#!/usr/bin/env python
"""
Data-generation toolkit for BIP Practical 1: *From Airfoil Geometry to Lift and Drag*.

Everything in this file is **provided** and should not be modified. What is yours:
the sampler, the bounds you choose inside the clamp, the allocation of the budget,
the train/validation split, and the model.

Design space (4 inputs, in this column order, always):

    X[:, 0] = m      maximum camber, PERCENT of chord      e.g. 4.0 for NACA 4412
    X[:, 1] = p      position of max camber, FRACTION      e.g. 0.4 for NACA 4412
    X[:, 2] = t      maximum thickness, PERCENT of chord   e.g. 12.0 for NACA 4412
    X[:, 3] = alpha  angle of attack, DEGREES

Note the mixed units: m and t are percentages, p is a fraction. This matches how the
NACA 4-digit designation is read aloud, and it makes an accidental column permutation
detectable (p is the only column that lives below 1). Re is fixed at 3e6 and Mach at 0;
neither is a design variable in this session.

Outputs:

    Y[:, 0] = CL
    Y[:, 1] = CD          <- raw. The AeroSurrogate contract does not ask for this.
    Y[:, 2] = CM
    conf    = analysis_confidence, NeuralFoil's own trustworthiness estimate in [0, 1]

--------------------------------------------------------------------------------
The contract
--------------------------------------------------------------------------------

Whatever you build must expose two methods, because that is all Practical 3 can
see of it on Wednesday:

    predict(X)      (N, 4) of [m, p, t, alpha]  ->  (N, 2) of [Cl, log10(Cd)]
    in_envelope(X)  (N, 4)                      ->  (N,) bool

Note the log10. The dataset does not do that transform for you.

`AeroSurrogate` is the abstract base. `train_surrogate` gives you a working
implementation of it in one line; `check_contract` tells you whether what you
built will survive Wednesday.

Requires:  neuralfoil            (pip install neuralfoil)
           pyLowOrder[NN]        (only for Part II -- Part I runs without it)
           scipy                 (optional; the envelope falls back if absent)
"""

from __future__ import annotations

import json
import time
import numpy as np

__all__ = [
    # -- constants ------------------------------------------------------------
    "CLAMP",
    "RE",
    "MACH",
    "COLUMNS",
    "BudgetExhausted",
    "ClampViolation",
    # -- Part I: generating the data ------------------------------------------
    "naca4_coordinates",
    "naca4_name",
    "assert_valid_queries",
    "NeuralFoilBudget",
    "generate_dataset",
    "save_campaign",
    "load_campaign",
    "plot_airfoil",
    # -- Part II: the contract ------------------------------------------------
    "AeroSurrogate",
    "Standardizer",
    "TargetTransform",
    "Envelope",
    "split_by_airfoil",
    "check_contract",
    # -- Part II: training ----------------------------------------------------
    "train_surrogate",
    "PyLOMSurrogate",
    "DEFAULT_ARCH",
    "DEFAULT_TRAINING_PARAMS",
]


# ----------------------------------------------------------------------------- #
#  Constants
# ----------------------------------------------------------------------------- #

#: Hard limits of the design space. Queries outside these bounds are refused
#: (free of charge -- a clamp violation is a bug, not a campaign decision).
CLAMP = {
    "m":     (0.0,  6.0),    # percent of chord
    "p":     (0.20, 0.70),   # fraction of chord
    "t":     (8.0,  18.0),   # percent of chord
    "alpha": (0.0,  10.0),   # degrees
}

RE = 3.0e6      #: Reynolds number, fixed for the whole session.
MACH = 0.0      #: Incompressible.

COLUMNS = ("m [%]", "p [-]", "t [%]", "alpha [deg]")


class BudgetExhausted(RuntimeError):
    """Raised when a request would exceed the remaining evaluation budget.

    The request is refused *atomically*: nothing is evaluated and nothing is
    spent. You still have `evaluator.remaining` evaluations left.
    """


class ClampViolation(ValueError):
    """Raised when a query falls outside the permitted design space."""


# ----------------------------------------------------------------------------- #
#  Geometry
# ----------------------------------------------------------------------------- #

def naca4_coordinates(m: float, p: float, t: float, n_points: int = 161) -> np.ndarray:
    """Analytic NACA 4-digit section, generated for *continuous* m, p, t.

    Built from the closed-form equations rather than the 4-digit name string,
    so the design space stays continuous instead of collapsing onto the integer
    digit grid.

    Parameters
    ----------
    m : float
        Maximum camber, percent of chord.
    p : float
        Chordwise position of maximum camber, fraction of chord.
    t : float
        Maximum thickness, percent of chord.
    n_points : int
        Total number of coordinate points returned (forced odd).

    Returns
    -------
    (n_points, 2) float array
        Coordinates in Selig order: upper-surface trailing edge, forward to the
        leading edge, then aft along the lower surface to the trailing edge.
        Chord is 1.0, leading edge at the origin. Trailing edge is closed.
    """
    mc = float(m) / 100.0
    tc = float(t) / 100.0
    pc = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    n_half = int(n_points) // 2 + 1
    beta = np.linspace(0.0, np.pi, n_half)
    x = 0.5 * (1.0 - np.cos(beta))                     # cosine spacing, LE clustered

    # Thickness distribution (closed trailing edge: -0.1036 rather than -0.1015)
    yt = 5.0 * tc * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )

    # Mean camber line, piecewise about x = p
    fore = x < pc
    yc = np.where(
        fore,
        mc / pc ** 2 * (2.0 * pc * x - x ** 2),
        mc / (1.0 - pc) ** 2 * ((1.0 - 2.0 * pc) + 2.0 * pc * x - x ** 2),
    )
    dyc = np.where(
        fore,
        2.0 * mc / pc ** 2 * (pc - x),
        2.0 * mc / (1.0 - pc) ** 2 * (pc - x),
    )
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack([xu, yu])[::-1]            # TE -> LE
    lower = np.column_stack([xl, yl])[1:]              # LE -> TE, LE not repeated
    coords = np.vstack([upper, lower])

    coords[0] = [1.0, 0.0]                             # snap TE exactly closed
    coords[-1] = [1.0, 0.0]
    return coords


def naca4_name(m: float, p: float, t: float) -> str:
    """Nearest 4-digit designation, for labelling plots only.

    The underlying geometry is continuous; this is a lossy label. Two different
    designs can share a name.
    """
    return "NACA %d%d%02d" % (round(m), round(p * 10), round(t))


# ----------------------------------------------------------------------------- #
#  Query validation
# ----------------------------------------------------------------------------- #

def assert_valid_queries(X: np.ndarray, clamp: dict = CLAMP) -> np.ndarray:
    """Check a query matrix before spending budget on it.

    Catches the three mistakes that cost groups the most: wrong shape, permuted
    columns, and out-of-clamp values.

    Parameters
    ----------
    X : (N, 4) array
        Columns in the order [m, p, t, alpha]. See module docstring for units.

    Returns
    -------
    (N, 4) float64 array
        The validated queries, as a contiguous float64 copy.

    Raises
    ------
    ValueError, ClampViolation
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(
            f"queries must be (N, 4) with columns {COLUMNS}, got shape {X.shape}"
        )
    if X.shape[0] == 0:
        raise ValueError("empty query matrix")
    if not np.all(np.isfinite(X)):
        bad = int(np.argmax(~np.all(np.isfinite(X), axis=1)))
        raise ValueError(f"non-finite value in query matrix, first at row {bad}")

    # Column-permutation heuristic: p is the only variable below 1.
    if np.max(X[:, 1]) > 1.0:
        raise ClampViolation(
            "column 1 (p) contains values above 1 -- are your columns in the "
            f"order {COLUMNS}? p is a fraction of chord, m and t are percentages."
        )

    for j, key in enumerate(("m", "p", "t", "alpha")):
        lo, hi = clamp[key]
        below = X[:, j] < lo - 1e-12
        above = X[:, j] > hi + 1e-12
        if below.any() or above.any():
            k = int(np.argmax(below | above))
            raise ClampViolation(
                f"{COLUMNS[j]} out of clamp [{lo}, {hi}] at row {k}: {X[k, j]:.4f}"
            )

    return np.ascontiguousarray(X)


# ----------------------------------------------------------------------------- #
#  Metered evaluator
# ----------------------------------------------------------------------------- #

class NeuralFoilBudget:
    """NeuralFoil behind a hard evaluation cap.

    The budget is the point of the exercise. Rules, all deliberate:

    1. **Charged per row.** NeuralFoil is vectorised over alpha, so one call
       covering fifty angles costs fifty, not one.
    2. **Atomic rejection.** A request larger than what remains is refused
       whole. Nothing is evaluated, nothing is spent.
    3. **No deduplication.** Ask for the same design twice, pay twice.
    4. **Low confidence costs full price.** NeuralFoil always returns an
       answer, so you are paying for the query, not for the answer being good.
    5. **Clamp violations are free** but raise -- they are bugs, not decisions.
    6. **Everything is logged**, including refused requests.

    Parameters
    ----------
    budget : int
        Hard cap on the number of evaluations.
    model_size : str
        NeuralFoil network size. One of "xxsmall" ... "xxxlarge". Generate at
        "medium"; the sealed test sets are refereed at the largest size.
    tag : str
        Label carried into the log and the saved metadata.
    clamp : dict
        Permitted design space. Leave at the default for anything a student
        touches. The sealed-set generator widens it deliberately, because sets
        B and C live outside the box the students are allowed to buy from.

    Examples
    --------
    >>> nfb = NeuralFoilBudget(budget=2000)             # doctest: +SKIP
    >>> out = nfb(m=4.0, p=0.4, t=12.0, alpha=[0, 2, 4])
    >>> nfb.remaining
    1997
    """

    def __init__(self, budget: int = 2000, model_size: str = "medium",
                 tag: str = "campaign", n_points: int = 161,
                 clamp: dict = CLAMP):
        self._budget = int(budget)
        self._spent = 0
        self.model_size = str(model_size)
        self.tag = str(tag)
        self.n_points = int(n_points)
        self.clamp = dict(clamp)
        self.log: list[dict] = []
        self._t0 = time.time()

    # -- accounting ----------------------------------------------------------

    @property
    def budget(self) -> int:
        """Total evaluations granted."""
        return self._budget

    @property
    def spent(self) -> int:
        """Evaluations consumed so far."""
        return self._spent

    @property
    def remaining(self) -> int:
        """Evaluations still available."""
        return self._budget - self._spent

    def check(self, n: int) -> None:
        """Raise `BudgetExhausted` if `n` more evaluations would overrun.

        Call this before a large campaign so it fails before any of it runs.
        """
        n = int(n)
        if n > self.remaining:
            self._log(kind="refused", n=n)
            raise BudgetExhausted(
                f"request for {n} evaluations, {self.remaining} remaining "
                f"(spent {self._spent} of {self._budget}). Nothing was spent."
            )

    def _log(self, **kw) -> None:
        kw["t"] = round(time.time() - self._t0, 3)
        kw["remaining_after"] = self.remaining
        self.log.append(kw)

    # -- evaluation ----------------------------------------------------------

    def _call_neuralfoil(self, coords: np.ndarray, alpha: np.ndarray) -> dict:
        """The single point of contact with the NeuralFoil library."""
        import neuralfoil as nf
        return nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alpha,
            Re=RE,
            model_size=self.model_size,
        )

    def __call__(self, m: float, p: float, t: float, alpha) -> dict:
        """Evaluate one geometry at one or more angles of attack.

        Parameters
        ----------
        m, p, t : float
            Scalar geometry. Percent, fraction, percent.
        alpha : float or array-like
            Angle(s) of attack in degrees. Cost equals `np.size(alpha)`.

        Returns
        -------
        dict
            Keys "CL", "CD", "CM", "analysis_confidence", each a 1-D array of
            length `np.size(alpha)`.
        """
        alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        n = alpha.size

        geom = np.column_stack([
            np.full(n, float(m)), np.full(n, float(p)),
            np.full(n, float(t)), alpha,
        ])
        assert_valid_queries(geom, self.clamp)   # free; raises on clamp violation

        self.check(n)                       # atomic: raises before spending

        coords = naca4_coordinates(m, p, t, n_points=self.n_points)
        raw = self._call_neuralfoil(coords, alpha)

        self._spent += n
        self._log(kind="spend", n=n, m=float(m), p=float(p), t=float(t),
                  alpha_min=float(alpha.min()), alpha_max=float(alpha.max()))

        return {
            "CL": np.asarray(raw["CL"], dtype=np.float64).reshape(n),
            "CD": np.asarray(raw["CD"], dtype=np.float64).reshape(n),
            "CM": np.asarray(raw["CM"], dtype=np.float64).reshape(n),
            "analysis_confidence": np.asarray(
                raw["analysis_confidence"], dtype=np.float64).reshape(n),
        }

    # -- reporting -----------------------------------------------------------

    def report(self) -> str:
        """One-line spend summary."""
        calls = sum(1 for e in self.log if e["kind"] == "spend")
        refused = sum(1 for e in self.log if e["kind"] == "refused")
        return (f"[{self.tag}] spent {self._spent}/{self._budget} "
                f"({self.remaining} left) in {calls} calls, "
                f"{refused} refused, model_size={self.model_size!r}")

    def metadata(self) -> dict:
        """Serialisable record of this evaluator's state."""
        return {
            "tag": self.tag,
            "budget": self._budget,
            "spent": self._spent,
            "remaining": self.remaining,
            "model_size": self.model_size,
            "n_points": self.n_points,
            "Re": RE,
            "mach": MACH,
            "clamp": {k: list(v) for k, v in self.clamp.items()},
            "columns": list(COLUMNS),
        }


# ----------------------------------------------------------------------------- #
#  Dataset assembly
# ----------------------------------------------------------------------------- #

def generate_dataset(evaluator: NeuralFoilBudget, queries: np.ndarray,
                     verbose: bool = True):
    """Run a campaign and return it as flat numpy arrays.

    Takes a flat (N, 4) query matrix rather than a shapes-by-angles grid, so no
    particular allocation of the budget is privileged by the interface. Rows
    sharing an identical geometry are grouped internally and evaluated in one
    vectorised call; this saves wall-clock, never budget.

    The whole campaign is checked against the remaining budget before any of it
    runs, so an over-large campaign fails without spending anything.

    Parameters
    ----------
    evaluator : NeuralFoilBudget
    queries : (N, 4) array
        Columns [m, p, t, alpha]. See module docstring for units.

    Returns
    -------
    X : (N, 4) float64
        The queries, in the order given. Feed straight to `predict`.
    Y : (N, 3) float64
        [CL, CD, CM].
    conf : (N,) float64
        NeuralFoil's analysis_confidence.
    """
    X = assert_valid_queries(queries, evaluator.clamp)
    n_total = X.shape[0]
    evaluator.check(n_total)

    Y = np.full((n_total, 3), np.nan, dtype=np.float64)
    conf = np.full(n_total, np.nan, dtype=np.float64)

    geoms, inverse = np.unique(X[:, :3], axis=0, return_inverse=True)
    for g, geom in enumerate(geoms):
        rows = np.flatnonzero(inverse == g)
        out = evaluator(geom[0], geom[1], geom[2], X[rows, 3])
        Y[rows, 0] = out["CL"]
        Y[rows, 1] = out["CD"]
        Y[rows, 2] = out["CM"]
        conf[rows] = out["analysis_confidence"]

    if verbose:
        print(f"{n_total} rows generated. {evaluator.report()}")
    return X, Y, conf


# ----------------------------------------------------------------------------- #
#  Persistence
# ----------------------------------------------------------------------------- #

def save_campaign(path: str, X: np.ndarray, Y: np.ndarray, conf: np.ndarray,
                  evaluator: NeuralFoilBudget, campaign_card: dict | None = None
                  ) -> None:
    """Write `<path>.npz` (arrays) and `<path>.json` (metadata + spend log).

    Arrays stay arrays; everything descriptive lives in the sidecar. The JSON is
    half of your deliverable -- the campaign card travels with the data.
    """
    stem = path[:-4] if path.endswith(".npz") else path
    np.savez_compressed(stem + ".npz", X=X, Y=Y, conf=conf)

    meta = evaluator.metadata()
    meta["n_rows"] = int(X.shape[0])
    meta["n_geometries"] = int(np.unique(X[:, :3], axis=0).shape[0])
    meta["campaign_card"] = campaign_card or {}
    meta["log"] = evaluator.log
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)


def load_campaign(path: str):
    """Inverse of `save_campaign`. Returns `(X, Y, conf, meta)`."""
    stem = path[:-4] if path.endswith(".npz") else path
    with np.load(stem + ".npz") as d:
        X, Y, conf = d["X"], d["Y"], d["conf"]
    try:
        with open(stem + ".json") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        meta = {}
    return X, Y, conf, meta


# ----------------------------------------------------------------------------- #
#  Convenience
# ----------------------------------------------------------------------------- #

def plot_airfoil(m: float, p: float, t: float, ax=None, **kw):
    """Draw a section to equal axes. Returns the axis."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2))
    c = naca4_coordinates(m, p, t)
    kw.setdefault("label", naca4_name(m, p, t))
    ax.plot(c[:, 0], c[:, 1], **kw)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    return ax


# ----------------------------------------------------------------------------- #
#  PART II -- The contract
# ----------------------------------------------------------------------------- #

"""
Everything below is for the afternoon. Part I needs none of it, and importing
this module does not require pyLOM or torch -- those are imported lazily, inside
`train_surrogate`, so the data-generation half works on a bare install.
"""

from abc import ABC, abstractmethod

EPS = 1e-12


class AeroSurrogate(ABC):
    """What Practical 3 imports. Implement these two methods and you are done.

    Wednesday's reinforcement-learning environment calls `predict` a few hundred
    thousand times and `in_envelope` exactly as often. Nothing else about your
    model is visible to it.
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) of [m, p, t, alpha]  ->  (N, 2) of [Cl, log10(Cd)].

        Must accept any finite input, including inputs far outside anything you
        trained on. Returning NaN or raising for out-of-range input breaks the
        RL environment. If you do not know the answer, say so through
        `in_envelope` -- not by refusing to answer.
        """

    @abstractmethod
    def in_envelope(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) -> (N,) bool. True where `predict` is worth believing.

        This is the single most important method you will write this week, and
        it is not the one you will spend the most time on.

        On Wednesday an agent will search your action space for the place where
        your surrogate promises the most lift for the least drag. If your model
        is confidently wrong somewhere, it will find that place, and the only
        defence available is this function returning False there.
        """

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        raise NotImplementedError("your surrogate must define save()")

    @classmethod
    def load(cls, path: str) -> "AeroSurrogate":
        raise NotImplementedError("your surrogate must define load()")

    # -- convenience ---------------------------------------------------------

    def predict_cd(self, X: np.ndarray) -> np.ndarray:
        """Cd on the natural scale, for plotting and for the reward function."""
        return 10.0 ** self.predict(X)[:, 1]

    def lift_to_drag(self, X: np.ndarray) -> np.ndarray:
        """Cl/Cd. What Wednesday's agent is actually trying to maximise."""
        out = self.predict(X)
        return out[:, 0] / np.maximum(10.0 ** out[:, 1], EPS)

    def envelope_score(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) -> (N,) float. Graded doubt. Small is safe, 1 is the edge.

        Optional. The default below derives it from `in_envelope`, so every
        surrogate that honours the contract has one and nothing needs changing.

        Override it if you can do better. Wednesday's last exercise penalises an
        agent in proportion to how far outside your envelope it has wandered,
        and a step function is a poor thing to penalise with: everything outside
        looks equally bad, so nothing pulls the agent back in. If your
        `in_envelope` is built on a distance -- and if it is a `pyLOM`
        `Envelope`, it is -- return that distance normalised by its threshold
        instead, and the penalty acquires a slope.
        """
        return np.where(np.asarray(self.in_envelope(X), dtype=bool), 0.0, 1.0)


# ----------------------------------------------------------------------------- #
#  Scaling
# ----------------------------------------------------------------------------- #

class Standardizer:
    """Zero mean, unit variance, per column.

    Fit on the training split only. Fitting on everything leaks the validation
    distribution into the model, which is a smaller sin than leaking the rows
    themselves but the same kind of sin.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, A: np.ndarray) -> "Standardizer":
        A = np.asarray(A, dtype=np.float64)
        self.mean_ = A.mean(axis=0)
        self.std_ = np.maximum(A.std(axis=0), EPS)
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (np.asarray(A, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, A: np.ndarray) -> np.ndarray:
        return np.asarray(A, dtype=np.float64) * self.std_ + self.mean_

    def fit_transform(self, A: np.ndarray) -> np.ndarray:
        return self.fit(A).transform(A)

    def state(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "Standardizer":
        obj = cls()
        obj.mean_ = np.asarray(s["mean"], dtype=np.float64)
        obj.std_ = np.asarray(s["std"], dtype=np.float64)
        return obj


class TargetTransform:
    """[Cl, Cd, Cm] as generated  <->  [Cl, log10(Cd)] as the contract wants.

    Provided, not applied. Whether to use it is a decision, and a group that
    trains on raw Cd will find out why it was here.
    """

    @staticmethod
    def forward(Y: np.ndarray) -> np.ndarray:
        """(N, 3) of [Cl, Cd, Cm]  ->  (N, 2) of [Cl, log10(Cd)]."""
        Y = np.asarray(Y, dtype=np.float64)
        return np.column_stack([Y[:, 0], np.log10(np.maximum(Y[:, 1], EPS))])

    @staticmethod
    def inverse(Z: np.ndarray) -> np.ndarray:
        """(N, 2) of [Cl, log10(Cd)]  ->  (N, 2) of [Cl, Cd]."""
        Z = np.asarray(Z, dtype=np.float64)
        return np.column_stack([Z[:, 0], 10.0 ** Z[:, 1]])


# ----------------------------------------------------------------------------- #
#  Envelope
# ----------------------------------------------------------------------------- #

class Envelope:
    """Two-test answer to "did I train anywhere near here?".

    **Test 1, the box.** Per-axis min and max of the training inputs, with an
    optional margin. Cheap, interpretable, and catches the failure that matters
    most on Wednesday -- an agent walking t out to 24% when nothing above 18%
    was ever seen.

    **Test 2, the neighbour distance.** The box says nothing about holes. A
    campaign that sampled t = 8 and t = 18 and nothing between has a box
    covering [8, 18] and no knowledge of the middle. So: standardise the input
    space, measure the distance to the nearest training point, and reject
    anything further away than a chosen quantile of the training set's own
    nearest-neighbour distances.

    `in_envelope` is the AND of the two. A point must be inside the box *and*
    close to something real.

    The quantile is the honesty dial. At 0.99 the envelope is generous and the
    agent gets more room to exploit; at 0.90 it is strict and may reject valid
    interpolation. Groups should pick a value and be able to say why.

    Backed by a KD-tree (`scipy.spatial.cKDTree`), with a chunked brute-force
    fallback if scipy is missing. This matters more than it looks: Practical 3
    calls this inside the RL loop, once per environment step, and brute force
    against 1600 training rows costs roughly 3 ms per 256-row batch -- about ten
    minutes per 200k steps, which is most of the session. The tree makes it
    negligible and is the difference between vectorised environments being
    usable on Wednesday and not.

    Not implemented via a POD projection residual, though Lecture 1 motivates it
    that way. In four dimensions with 2000 samples the residual of a linear
    subspace projection is close to meaningless -- the inputs already span the
    space. Nearest-neighbour distance is the version of the same idea that
    survives contact with a low-dimensional design space. Worth saying aloud,
    because a sharp student will ask.
    """

    def __init__(self, quantile: float = 0.99, margin: float = 0.0,
                 calibration: str = "uniform", n_calibration: int = 4000,
                 seed: int = 0) -> None:
        self.quantile = float(quantile)
        self.margin = float(margin)
        self.calibration = str(calibration)
        self.n_calibration = int(n_calibration)
        self.seed = int(seed)
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None
        self.scaler_: Standardizer | None = None
        self.Xs_: np.ndarray | None = None
        self.threshold_: float | None = None
        self._tree = None

    # -- fitting -------------------------------------------------------------

    def fit(self, X: np.ndarray, calibrate_on: np.ndarray | None = None
            ) -> "Envelope":
        """Fit the box and the distance threshold on the training inputs.

        `calibrate_on` sets what the threshold is calibrated against. Leave it
        None and `calibration` decides:

            "uniform" (default)  a fresh uniform sample drawn inside the fitted
                                 box, measuring how far a typical *query* sits
                                 from the training data
            "train"              distance from each training point to its
                                 nearest other training point

        "train" is the obvious choice and it is wrong whenever the campaign is
        not uniformly dense. It measures how tightly the training set clusters
        with itself, which is a property of the sampling strategy rather than of
        the queries. Load a quarter of the shapes onto the clamp faces and the
        train-to-train distances collapse -- the boundary points are packed
        together -- while an interior query still sits far from all of them. The
        threshold then comes out too small and rejects a slab of the interior
        for no reason.

        Measured on the two instructor campaigns:

            campaign     train-train   uniform-in-box   ratio
            reference        0.294         0.370         1.26
            exemplar         0.242         0.387         1.60

        The exemplar covers more of the space and still has the *smaller*
        train-train distance. Calibrating on it rejected 11.8% of the interior.

        Held-out data from the same campaign is not a fix either: it inherits
        the campaign's sampling bias. Uniform inside the box is the honest
        reference distribution, because it is what Wednesday's agent queries.
        """
        X = np.asarray(X, dtype=np.float64)
        span = X.max(axis=0) - X.min(axis=0)
        self.lo_ = X.min(axis=0) - self.margin * span
        self.hi_ = X.max(axis=0) + self.margin * span

        self.scaler_ = Standardizer().fit(X)
        self.Xs_ = np.ascontiguousarray(self.scaler_.transform(X))
        self._build_tree()

        if calibrate_on is not None:
            d = self.nn_distance(np.asarray(calibrate_on, dtype=np.float64))
        elif self.calibration == "uniform":
            rng = np.random.default_rng(self.seed)
            probe = rng.uniform(self.lo_, self.hi_,
                                size=(self.n_calibration, X.shape[1]))
            d = self.nn_distance(probe)
        elif self.calibration == "train":
            # k=2 because k=1 is the point itself.
            if self._tree is not None:
                d = self._tree.query(self.Xs_, k=2)[0][:, 1]
            else:
                d = self._brute_nn(self.Xs_, exclude_self=True)
        else:
            raise ValueError("calibration must be 'uniform' or 'train'")

        self.threshold_ = float(np.quantile(d, self.quantile))
        return self

    def _build_tree(self) -> None:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._tree = None
            return
        self._tree = cKDTree(self.Xs_)

    # -- queries -------------------------------------------------------------

    def _brute_nn(self, Q: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """Fallback when scipy is unavailable. Chunked, to bound memory."""
        out = np.empty(Q.shape[0], dtype=np.float64)
        step = 4096
        for i in range(0, Q.shape[0], step):
            block = Q[i:i + step]
            d = np.sqrt(np.maximum(
                (block ** 2).sum(1)[:, None]
                - 2.0 * block @ self.Xs_.T
                + (self.Xs_ ** 2).sum(1)[None, :], 0.0))
            if exclude_self:
                np.fill_diagonal(d[:, i:i + block.shape[0]], np.inf)
            out[i:i + step] = d.min(axis=1)
        return out

    def nn_distance(self, X: np.ndarray) -> np.ndarray:
        """Scaled distance from each query to the nearest training point.

        Useful on its own -- plotting this against prediction error on the
        sealed sets is the most direct evidence that the envelope means
        something.
        """
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        if self._tree is not None:
            return self._tree.query(Xs, k=1)[0]
        return self._brute_nn(Xs)

    def in_box(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.all((X >= self.lo_) & (X <= self.hi_), axis=1)

    def near_data(self, X: np.ndarray) -> np.ndarray:
        return self.nn_distance(X) <= self.threshold_

    def score(self, X: np.ndarray) -> np.ndarray:
        """Graded version of `near_data`: distance to data over the threshold.

        Below 1 the query sits closer to the training set than the calibration
        quantile; above 1 it is further. Same test as `near_data`, without the
        step function.

        Practical 3 needs the graded form. A boolean envelope gives a reward
        function a cliff -- an agent one step outside it feels the same penalty
        as an agent halfway to nowhere, and the gradient that would have walked
        it back inside does not exist. `score` restores that gradient.

        Deliberately *not* the same test as `__call__`, which is the AND of the
        box and the distance. A point can be far outside the box and still score
        below 1 if the box is long and thin in that direction. Use `__call__`
        when you want the honest yes/no, `score` when you want something to
        differentiate. Penalising `max(score - 1, 0)` and rejecting on
        `not in_envelope` are complementary, not redundant.
        """
        return self.nn_distance(X) / max(float(self.threshold_), EPS)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.in_box(X) & self.near_data(X)

    # -- persistence ---------------------------------------------------------

    def state(self) -> dict:
        return {
            "quantile": self.quantile,
            "margin": self.margin,
            "calibration": self.calibration,
            "n_calibration": self.n_calibration,
            "seed": self.seed,
            "lo": self.lo_.tolist(),
            "hi": self.hi_.tolist(),
            "scaler": self.scaler_.state(),
            "threshold": self.threshold_,
        }

    @classmethod
    def from_state(cls, s: dict, Xtrain: np.ndarray) -> "Envelope":
        """Rebuild from saved state plus the training inputs.

        The training inputs are needed because the tree is not serialisable in
        any useful way -- it is rebuilt on load, which takes milliseconds.
        """
        obj = cls(quantile=s["quantile"], margin=s["margin"],
                  calibration=s.get("calibration", "train"),
                  n_calibration=s.get("n_calibration", 4000),
                  seed=s.get("seed", 0))
        obj.lo_ = np.asarray(s["lo"], dtype=np.float64)
        obj.hi_ = np.asarray(s["hi"], dtype=np.float64)
        obj.scaler_ = Standardizer.from_state(s["scaler"])
        obj.Xs_ = np.ascontiguousarray(obj.scaler_.transform(Xtrain))
        obj.threshold_ = float(s["threshold"])
        obj._build_tree()
        return obj


# ----------------------------------------------------------------------------- #
#  Splitting
# ----------------------------------------------------------------------------- #

def split_by_airfoil(X: np.ndarray, frac: float = 0.2, seed: int = 0):
    """Group-aware split. Every alpha of a given airfoil lands on one side.

    Returns (train_idx, val_idx).

    A random row split reports a validation error that is measuring
    interpolation between two points you already own, not generalisation. The
    unit of independence in this dataset is the *airfoil*, not the row.

    Provided because it is the correct answer, not because using it is
    mandatory. A group is free to split randomly, and the gap between their
    validation curve and their leaderboard is then the lesson.
    """
    X = np.asarray(X, dtype=np.float64)
    geoms, inverse = np.unique(X[:, :3], axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(geoms.shape[0])
    n_val = max(1, int(round(frac * geoms.shape[0])))
    val_geoms = set(order[:n_val].tolist())
    is_val = np.array([g in val_geoms for g in inverse])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


# ----------------------------------------------------------------------------- #
#  PART II -- Training
# ----------------------------------------------------------------------------- #

"""
`train_surrogate` wraps pyLOM's MLP so you can have a working model in one line
and spend your afternoon on the parts of this session that matter -- the
campaign, the split, the envelope, and finding out where your own model lies.

Three things it decides for you, each tied to a lesson, each overridable:

    target="log10"   trains on log10(Cd). Pass "raw" to train on Cd and find
                     out why the contract asks for the logarithm.
    split="airfoil"  every angle of a geometry lands on one side of the
                     train/validation line. Pass "random" for the version
                     everyone writes by default, then compare your validation
                     curve against your leaderboard score.
    scalers fitted on the training split only. Fitting on everything leaks the
    validation distribution into the model.

Requires:  pip install pyLowOrder[NN]
"""

import os


#: Sensible starting point for 2000 rows on a Colab CPU. Roughly 20 s.
#: Not tuned. Tuning it is a legitimate use of your afternoon; it is not the
#: most valuable use of your afternoon.
DEFAULT_TRAINING_PARAMS = {
    "epochs": 300,
    "lr": 1e-3,
    "lr_gamma": 0.98,
    "lr_scheduler_step": 20,
    "batch_size": 64,
    "print_rate_epoch": 25,
    "num_workers": 0,
}

DEFAULT_ARCH = {
    "hidden_size": 96,
    "n_layers": 3,
    "p_dropouts": 0.0,
}


# ----------------------------------------------------------------------------- #
#  Training entry point
# ----------------------------------------------------------------------------- #

def train_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    target: str = "log10",
    split: str = "airfoil",
    val_frac: float = 0.2,
    seed: int = 0,
    envelope_quantile: float = 0.99,
    envelope_calibration: str = "uniform",
    arch: dict | None = None,
    training_params: dict | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> "PyLOMSurrogate":
    """Train a pyLOM MLP on a campaign and return something Wednesday can use.

    Parameters
    ----------
    X : (N, 4)
        [m, p, t, alpha], straight out of `campaign.npz`.
    Y : (N, 3)
        [Cl, Cd, Cm], straight out of `campaign.npz`. Cm is not used.
    target : {"log10", "raw"}
        Whether to train on log10(Cd) or on Cd. The contract wants log10.
    split : {"airfoil", "random"}
        Group-aware or naive. See the module docstring.
    envelope_quantile : float
        How generous `in_envelope` is. Lower is stricter.
    envelope_calibration : {"uniform", "train"}
        What the distance threshold is calibrated against. See `Envelope.fit`.
        "train" is the naive choice and mis-calibrates badly whenever the
        campaign is not uniformly dense.
    arch, training_params : dict
        Merged over `DEFAULT_ARCH` / `DEFAULT_TRAINING_PARAMS`.

    Returns
    -------
    PyLOMSurrogate
    """
    import torch
    import pyLOM
    import pyLOM.NN

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X must be (N, 4), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"Y must be (N, 3) of [Cl, Cd, Cm], got {Y.shape}")
    if target not in ("log10", "raw"):
        raise ValueError("target must be 'log10' or 'raw'")

    Z = (TargetTransform.forward(Y) if target == "log10"
         else np.column_stack([Y[:, 0], Y[:, 1]]))

    # -- split ---------------------------------------------------------------
    if split == "airfoil":
        tr, va = split_by_airfoil(X, frac=val_frac, seed=seed)
    elif split == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(X.shape[0])
        n_val = int(round(val_frac * X.shape[0]))
        va, tr = perm[:n_val], perm[n_val:]
    else:
        raise ValueError("split must be 'airfoil' or 'random'")

    if verbose:
        n_gtr = np.unique(X[tr, :3], axis=0).shape[0]
        n_gva = np.unique(X[va, :3], axis=0).shape[0]
        print(f"train {tr.size} rows / {n_gtr} geometries    "
              f"val {va.size} rows / {n_gva} geometries    split={split}")

    # -- pyLOM datasets ------------------------------------------------------
    # The scalers are fitted by whichever Dataset is constructed first, and
    # reused by the second. Training set first, therefore, and not by accident.
    dev = pyLOM.NN.select_device(device) if device else pyLOM.NN.select_device()

    input_scaler = pyLOM.NN.MinMaxScaler()
    output_scaler = pyLOM.NN.MinMaxScaler()

    def _make(idx, in_s, out_s):
        return pyLOM.NN.Dataset(
            variables_out=(Z[idx, 0:1], Z[idx, 1:2]),
            variables_in=X[idx],
            inputs_scaler=in_s,
            outputs_scaler=out_s,
            snapshots_by_column=False,
        )

    td_train = _make(tr, input_scaler, output_scaler)
    td_val = _make(va, input_scaler, output_scaler)

    # -- model ---------------------------------------------------------------
    a = {**DEFAULT_ARCH, **(arch or {})}
    tp = {**DEFAULT_TRAINING_PARAMS, **(training_params or {})}
    tp.setdefault("loss_fn", torch.nn.MSELoss())
    tp.setdefault("optimizer_class", torch.optim.Adam)
    tp["device"] = dev

    sample_in, sample_out = td_train[0]
    model = pyLOM.NN.MLP(
        input_size=sample_in.shape[0],
        output_size=sample_out.shape[0],
        hidden_size=a["hidden_size"],
        n_layers=a["n_layers"],
        p_dropouts=a["p_dropouts"],
        device=dev,
        seed=seed,
    )

    pipeline = pyLOM.NN.Pipeline(
        train_dataset=td_train,
        test_dataset=td_val,
        model=model,
        training_params=tp,
    )
    logs = pipeline.run()

    # -- envelope ------------------------------------------------------------
    env = Envelope(quantile=envelope_quantile,
                   calibration=envelope_calibration, seed=seed).fit(X[tr])

    surrogate = PyLOMSurrogate(
        model=pipeline.model,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        envelope=env,
        Xtrain=X[tr],
        target=target,
        logs=logs,
        meta={"split": split, "seed": seed, "arch": a,
              "envelope_calibration": envelope_calibration,
              "n_train": int(tr.size), "n_val": int(va.size)},
    )
    # Kept in memory only, not saved. Lets you score the model on its own
    # held-out rows in contract units, which is the honest way to see what your
    # validation number is worth -- see `holdout_mae`.
    surrogate.train_idx = tr
    surrogate.val_idx = va
    return surrogate


# ----------------------------------------------------------------------------- #
#  The shipped object
# ----------------------------------------------------------------------------- #

class PyLOMSurrogate(AeroSurrogate):
    """A trained pyLOM MLP wearing the AeroSurrogate contract.

    Holds the model, both scalers, and the envelope. `predict` returns an answer
    for any finite input, including inputs nowhere near the training data --
    that is required, because Wednesday's agent will ask, and raising would kill
    the environment. Whether the answer is worth anything is what `in_envelope`
    is for.
    """

    train_idx = None      # set by train_surrogate; not restored by load()
    val_idx = None

    def __init__(self, model, input_scaler, output_scaler,
                 envelope, Xtrain, target="log10", logs=None, meta=None):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.envelope = envelope
        self.Xtrain = np.ascontiguousarray(np.asarray(Xtrain, dtype=np.float64))
        self.target = target
        self.logs = logs or {}
        self.meta = meta or {}

    # -- contract ------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) -> (N, 2) of [Cl, log10(Cd)].

        Scale the inputs, run the network, unscale the outputs. `MinMaxScaler`
        splits a 2-D array into one variable per column and re-stacks the result,
        so a plain (N, 4) array in gives a plain (N, 4) array back -- the same
        idiom as `example_MLP_DLR_airfoil.py`.
        """
        import torch

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xs = np.asarray(self.input_scaler.transform(X), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(
                torch.tensor(Xs, device=self.model.device)).cpu().numpy()

        Z = np.asarray(self.output_scaler.inverse_transform(
            np.asarray(preds, dtype=np.float64)), dtype=np.float64)
        Z = Z.reshape(X.shape[0], 2)

        if self.target == "raw":
            # Trained on Cd; the contract still wants log10(Cd).
            Z = np.column_stack([Z[:, 0], np.log10(np.maximum(Z[:, 1], 1e-12))])
        return Z

    def in_envelope(self, X: np.ndarray) -> np.ndarray:
        return self.envelope(np.atleast_2d(np.asarray(X, dtype=np.float64)))

    def envelope_score(self, X: np.ndarray) -> np.ndarray:
        """Graded doubt, from the envelope's own distance. See `AeroSurrogate`.

        Note this is the distance test alone, not the box test. `in_envelope`
        remains the AND of the two and stays the authority on whether an answer
        is worth believing.
        """
        return self.envelope.score(np.atleast_2d(np.asarray(X, dtype=np.float64)))

    # -- diagnostics ---------------------------------------------------------

    def holdout_mae(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """MAE on this model's own validation rows, in contract units.

        The loss curve `train_surrogate` prints is a mean square error on
        MinMaxScaled targets, which is not comparable to anything on the
        leaderboard. This is: the same metric, the same units, computed on the
        rows the model was not trained on.

        Pass the full campaign `X` and `Y`; the split indices are taken from the
        model. Only available on a freshly trained surrogate -- `load` does not
        restore them, because the campaign is not saved with the model.

        The number to look at is not this on its own but the ratio between it
        and the leaderboard. A model whose held-out estimate says one thing and
        whose sealed score says something three times worse has not been
        measuring generalisation.
        """
        if getattr(self, "val_idx", None) is None:
            raise RuntimeError(
                "no split indices on this surrogate -- holdout_mae only works "
                "on a freshly trained model, not a reloaded one")
        Z = TargetTransform.forward(Y)[self.val_idx]
        P = self.predict(np.asarray(X, dtype=np.float64)[self.val_idx])
        return {"mae_Cl": float(np.mean(np.abs(P[:, 0] - Z[:, 0]))),
                "mae_log10Cd": float(np.mean(np.abs(P[:, 1] - Z[:, 1])))}

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Write a directory containing everything needed to reload.

            <path>/model.pth        pyLOM MLP weights
            <path>/scalers/         both MinMaxScalers, as JSON
            <path>/envelope.npz     envelope state + training inputs
            <path>/meta.json        target convention, scaler conventions, logs
        """
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "scalers"), exist_ok=True)

        self.model.save(os.path.join(path, "model.pth"))
        self.input_scaler.save(os.path.join(path, "scalers", "input.json"))
        self.output_scaler.save(os.path.join(path, "scalers", "output.json"))

        np.savez_compressed(
            os.path.join(path, "envelope.npz"),
            Xtrain=self.Xtrain,
            state=np.frombuffer(
                json.dumps(self.envelope.state()).encode("utf-8"), dtype=np.uint8),
        )
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump({
                "target": self.target,
                "meta": self.meta,
                "logs": {k: list(map(float, v)) for k, v in self.logs.items()
                         if hasattr(v, "__iter__")},
            }, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PyLOMSurrogate":
        import pyLOM
        import pyLOM.NN

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)

        model = pyLOM.NN.MLP.load(os.path.join(path, "model.pth"))
        input_scaler = pyLOM.NN.MinMaxScaler.load(
            os.path.join(path, "scalers", "input.json"))
        output_scaler = pyLOM.NN.MinMaxScaler.load(
            os.path.join(path, "scalers", "output.json"))

        with np.load(os.path.join(path, "envelope.npz")) as d:
            Xtrain = d["Xtrain"]
            state = json.loads(bytes(d["state"]).decode("utf-8"))
        env = Envelope.from_state(state, Xtrain)

        return cls(model=model, input_scaler=input_scaler,
                   output_scaler=output_scaler,
                   envelope=env, Xtrain=Xtrain, target=meta["target"],
                   logs=meta.get("logs"), meta=meta.get("meta"))


# ----------------------------------------------------------------------------- #
#  Contract check
# ----------------------------------------------------------------------------- #

def check_contract(surrogate, verbose: bool = True) -> bool:
    """Would this survive Wednesday? Run it before you submit.

    Checks the four things that would otherwise fail in front of the room:
    output shape, finiteness far outside the envelope, `in_envelope`'s return
    type, and that `predict` is deterministic across calls.

    Passing this does not mean your surrogate is good. It means it is a valid
    surrogate. Those are different claims and only one of them is checkable
    from inside your own notebook.
    """
    ok = True
    say = print if verbose else (lambda *a, **k: None)

    probe = np.array([[3.0, 0.45, 12.0, 5.0],
                      [1.0, 0.30, 9.0, 2.0]])
    Z = np.asarray(surrogate.predict(probe))
    if Z.shape != (2, 2):
        say(f"FAIL  predict returned {Z.shape}; the contract is (N, 2) of "
            "[Cl, log10(Cd)]")
        ok = False

    # Wednesday's agent will ask about places you never sampled. Answering is
    # mandatory; being right is not. Doubt belongs in in_envelope.
    wild = np.array([[9.0, 0.85, 30.0, 25.0],
                     [0.0, 0.20, 4.0, -5.0]])
    if not np.all(np.isfinite(np.asarray(surrogate.predict(wild)))):
        say("FAIL  predict returned non-finite values far outside the envelope. "
            "It must always answer -- express doubt through in_envelope.")
        ok = False

    e = np.asarray(surrogate.in_envelope(probe))
    if e.shape != (2,) or e.dtype != bool:
        say(f"FAIL  in_envelope returned {e.shape} {e.dtype}, want (N,) bool")
        ok = False

    if not np.allclose(Z, np.asarray(surrogate.predict(probe)), atol=1e-9):
        say("FAIL  predict is not deterministic -- is the model still in "
            "training mode?")
        ok = False

    say("CONTRACT OK -- ready for Wednesday" if ok else "fix the above")
    return ok


# ----------------------------------------------------------------------------- #
#  Self-test  (python P1_ancillary.py)
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    c = naca4_coordinates(4.0, 0.4, 12.0)
    print("NACA 4412 coordinates:", c.shape)
    print("  closed TE:", np.allclose(c[0], [1, 0]) and np.allclose(c[-1], [1, 0]))
    # Surface points are offset along the local normal, so the upper surface
    # creeps a few 1e-4 ahead of the leading edge. Expected, not a defect.
    print(f"  x range: [{c[:, 0].min():+.2e}, {c[:, 0].max():.4f}]")

    sym = naca4_coordinates(0.0, 0.4, 12.0)
    n = sym.shape[0] // 2
    print("  symmetric NACA 0012:",
          np.allclose(sym[:n, 1], -sym[:0:-1, 1][:n], atol=1e-12))

    # thickness check: max(yu - yl) should equal t/100
    up, lo = sym[:n + 1][::-1], sym[n:]
    print("  max thickness:", round(float(np.max(up[:, 1] - lo[:, 1])), 4))

    # m = 0 degeneracy: p has no effect
    a = naca4_coordinates(0.0, 0.25, 12.0)
    b = naca4_coordinates(0.0, 0.65, 12.0)
    print("  (m=0, p) degeneracy present:", np.allclose(a, b))

    try:
        assert_valid_queries(np.array([[12.0, 4.0, 0.4, 2.0]]))
    except ClampViolation as e:
        print("permutation caught:", str(e)[:60], "...")

    try:
        assert_valid_queries(np.array([[4.0, 0.4, 12.0, 14.0]]))
    except ClampViolation as e:
        print("clamp caught:", str(e)[:60], "...")

    # -- Part II ----------------------------------------------------------
    rng = np.random.default_rng(0)
    Xd = np.column_stack([rng.uniform(*CLAMP[k], 400) for k in
                          ("m", "p", "t", "alpha")])
    tr, va = split_by_airfoil(Xd, 0.2, seed=1)
    gtr = {tuple(r) for r in np.unique(Xd[tr, :3], axis=0).tolist()}
    gva = {tuple(r) for r in np.unique(Xd[va, :3], axis=0).tolist()}
    print("split leaks no geometry:", len(gtr & gva) == 0)

    env = Envelope(0.99).fit(Xd[tr])
    print("envelope accepts training data:", env(Xd[tr]).mean() > 0.98)
    print("envelope rejects t = 30%:",
          not env(np.array([[3.0, 0.45, 30.0, 5.0]]))[0])
    s_in = env.score(Xd[va]).mean()   # held out: distance to train is not zero
    s_out = env.score(np.array([[3.0, 0.45, 30.0, 5.0]]))[0]
    print(f"envelope score: mean on held-out {s_in:.3f}, at t = 30% {s_out:.1f}")
    print("score is graded and monotone:",
          bool(s_in < 1.0 < s_out)
          and env.score(np.array([[3.0, 0.45, 22.0, 5.0]]))[0] < s_out)
    naive = Envelope(0.99, calibration="train").fit(Xd[tr])
    print(f"threshold uniform {env.threshold_:.3f} vs train-train "
          f"{naive.threshold_:.3f}")

    Yd = np.column_stack([rng.normal(size=400), rng.uniform(.005, .05, 400),
                          rng.normal(size=400)])
    Z = TargetTransform.forward(Yd)
    print("target transform round-trips:",
          np.allclose(TargetTransform.inverse(Z)[:, 1], Yd[:, 1]))
