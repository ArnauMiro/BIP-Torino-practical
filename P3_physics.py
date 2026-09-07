#!/usr/bin/env python
"""
READ THIS FIRST. IT IS A MOCK.
==============================
SPOILER WARNING, if you are a student and it is still Wednesday morning: this
file explains the trick you are about to walk into. Nothing here is hidden from
you and nothing is encrypted -- we would rather be honest and ask you to wait
than pretend. Close it, run the practical, then come back. If you do read it,
say so in the debrief; noticing is worth more than being surprised.

The drag correction in this file is NOT physics. It is a teaching device, and
it must be presented as one. Nobody should leave the room
believing that `delta_cd` is a published correlation, because it is not: the
functional form is invented and its magnitude was chosen to make a number on a
projector legible.

What we would like to have is a high-fidelity solver -- RANS, or a wind tunnel
-- disagreeing with the cheap model the surrogates were trained on. We cannot
run one inside a two-hour practical. So we mock the disagreement instead:

    P1 labels (Monday)      NeuralFoil, model_size="medium"
    P3 referee (Wednesday)  NeuralFoil, model_size="xxxlarge"   +   delta_cd

The size change is a genuine fidelity step. `delta_cd` is the amplifier that
makes that step large enough to see from the back of the room. Together they
stand in for "a better model of reality than the one that trained you", which
is the only claim being made.

WHY THE PENALTY SITS WHERE IT SITS
--------------------------------------------------------------
NeuralFoil reports `analysis_confidence`, its own estimate of whether it knows
what it is talking about. Measured at p = 0.40, alpha = 5, model_size
="xxxlarge", over the P1 clamp:

    m \\ t      8     10     12     15     18
      0     0.97   0.97   0.96   0.96   0.96
      2     0.98   0.97   0.96   0.95   0.95
      4     0.67   0.87   0.94   0.93   0.88
      6     0.47   0.50   0.56   0.59   0.61

Confidence halves in the thin, highly-cambered corner. The high-fidelity model
is telling us, unprompted, that this is the region it cannot resolve -- thin
NACA 4-digit sections have a leading-edge radius going as t^2, so a thin nose
with strong camber produces a suction peak and a separation behaviour that a
fast model handles poorly. That much is real.

Two things follow, and they are the whole practical:

    1. The unconstrained L/D optimum over the clamp sits at m = 6, t ~ 9 --
       inside the low-confidence corner. The naive answer is exactly the answer
       the model is least sure about.

    2. `medium` does NOT report the same doubt. At m = 6, t = 8 it returns 0.89
       where `xxxlarge` returns 0.47. So the cheap model is confidently wrong,
       and the students' campaigns inherited that confidence: across
       `P3_clean`, mean conf is 0.932 at m >= 5% against 0.961 at m <= 2%, a
       correlation with camber of only -0.51. The warning is in every group's
       `conf` column, far too weak for anyone to act on. That is the debrief.

So the placement of the penalty is empirical -- `severity` correlates -0.76
with that confidence map -- while its size is a choice. Say both.

CALIBRATION
-----------
`K_BOOST = 2.5e-3` was chosen, not fitted. Against a surrogate trained on clean
`medium` data, it produces:

    surrogate's recommended section    m = 6.0, t = 9.0, promising L/D 195
    what the referee says it delivers                            L/D 150
    promise gap                                                   1.30
    true optimum, which moved                m = 6.0, t = 11.5,  L/D 184

The agent overpromises by 30% and gives away 18% of the achievable L/D by
choosing a section 2.5 points too thin. Other values, if the room needs a
different number: 1.5e-3 gives a gap of 1.20, 4.0e-3 gives 1.44.

Note what the envelope does here: nothing. The lie is *inside* the training
data, so `in_envelope` returns True and the rung-3 penalty never fires. An
envelope tells you when you have left your data. It has no opinion on your data
being wrong in the first place. Only spending referee calls recovers this,
which is the argument for rung 4.
"""

from __future__ import annotations

import numpy as np

import P1_ancillary as bip

__all__ = [
    "K_BOOST", "S_ONSET", "M_REF", "T_REF",
    "severity", "delta_cd", "Referee", "pick_referee_size",
    "promise_gap", "DISCLOSURE", "print_disclosure",
]

#: Amplifier on the mocked high-fidelity drag rise. A teaching knob, not a
#: physical constant. See CALIBRATION above before changing it.
K_BOOST = 2.5e-3

#: Severity below which the booster adds nothing. Fixed at 1 so that the
#: reference section (m = 6%, t = 12%) sits exactly on the onset.
S_ONSET = 1.0

M_REF = 6.0     # camber [%] normalising the severity -- the clamp maximum
T_REF = 12.0    # thickness [%] normalising it -- a conventional section

#: Referee fidelity ladder, best first. Probed at run time because not every
#: NeuralFoil build ships every size.
REFEREE_SIZES = ("xxxlarge", "xxlarge", "xlarge", "large", "medium")


def severity(X: np.ndarray) -> np.ndarray:
    """(N, >=3) -> (N,). Camber over a leading-edge-radius proxy.

    The NACA 4-digit leading-edge radius goes as t^2, so `(t / T_REF) ** 2`
    stands in for how blunt the nose is, and camber over that is a crude index
    of how hard the suction peak has to work. Normalised so that S = 1 at
    m = 6%, t = 12%.

    Geometry only -- deliberately. If severity depended on Cl the students could
    reconstruct it from their own surrogate, and the whole point is that this
    effect is invisible from the training data.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return (X[:, 0] / M_REF) / np.maximum(X[:, 2] / T_REF, bip.EPS) ** 2


def delta_cd(X: np.ndarray, k: float = K_BOOST) -> np.ndarray:
    """(N, >=3) -> (N,). The mocked high-fidelity drag rise. Never negative.

    Quadratic in the excess severity, so the penalty switches on smoothly at
    S = 1 rather than stepping. The exponent is a choice; onset-plus-growth is
    the qualitative behaviour being imitated.
    """
    return k * np.maximum(0.0, severity(X) - S_ONSET) ** 2


def pick_referee_size(verbose: bool = True) -> str:
    """First entry of REFEREE_SIZES the installed NeuralFoil actually accepts.

    Same probe as `P1_gen_sealed_data`. Hard-coding "xxxlarge" fails at the
    moment of use, in front of everyone; probing fails at import, quietly.
    """
    probe = np.array([[4.0, 0.4, 12.0, 5.0]])
    for size in REFEREE_SIZES:
        try:
            ev = bip.NeuralFoilBudget(budget=1, model_size=size, tag="probe")
            bip.generate_dataset(ev, probe, verbose=False)
            return size
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                print(f"  model_size={size!r} unavailable ({type(exc).__name__})")
    raise RuntimeError("no referee model_size available")


class Referee:
    """Wednesday's judge: high-fidelity NeuralFoil plus the mocked booster.

    Deliberately unmetered. The referee is not a resource the students are
    spending, it is the thing that tells them whether they were lied to, and a
    budget on it would only teach them to check less. What IS metered is
    surrogate calls, because the evaluation count is half the RL-versus-DE
    argument -- see `P3_ancillary.CallCounter`.

    Enforces the P1 clamp, which with run-1 action bounds set to that same
    clamp makes it a free assertion: anything the referee refuses is a
    parameterizer clipping bug, not a student mistake.
    """

    def __init__(self, k: float = K_BOOST, model_size: str | None = None,
                 clamp: dict | None = None, verbose: bool = False):
        self.k = float(k)
        self.model_size = model_size or pick_referee_size(verbose=verbose)
        self.clamp = clamp or bip.CLAMP
        self._ev = bip.NeuralFoilBudget(budget=10 ** 9, model_size=self.model_size,
                                        tag="referee", clamp=self.clamp)

    def __call__(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 4) -> (Y, conf) with Y = [Cl, Cd] and Cd already boosted."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X, Y, conf = bip.generate_dataset(self._ev, X, verbose=False)
        Y = Y.copy()
        Y[:, 1] = Y[:, 1] + delta_cd(X, self.k)
        return Y, conf

    def lift_to_drag(self, X: np.ndarray) -> np.ndarray:
        """(N, 4) -> (N,). The number a design is finally judged on."""
        Y, _ = self(X)
        return Y[:, 0] / np.maximum(Y[:, 1], bip.EPS)

    def calls(self) -> int:
        return int(self._ev.spent)


def promise_gap(surrogate, X: np.ndarray, referee: Referee | None = None
                ) -> dict:
    """How much better the surrogate thinks a design is than it really is.

    Returns the promised L/D, the delivered L/D, their ratio, and -- because
    students conflate the two -- the *regret*, which is what the design gives up
    against the best the referee can find nearby. A surrogate can have a promise
    gap of 1.0 and still pick a poor design, and it can overpromise wildly on a
    design that happens to be fine.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    ref = referee or Referee()
    promised = np.asarray(surrogate.lift_to_drag(X), dtype=np.float64)
    delivered = ref.lift_to_drag(X)
    return {
        "promised": promised,
        "delivered": delivered,
        "gap": promised / np.maximum(delivered, bip.EPS),
        "in_envelope": np.asarray(surrogate.in_envelope(X), dtype=bool),
        "referee_model_size": ref.model_size,
        "k_boost": ref.k,
    }


DISCLOSURE = """\
------------------------------------------------------------------------
WHAT WE DID, AND WHY
------------------------------------------------------------------------
Your surrogate was trained on NeuralFoil at model_size="medium".
Today's referee is NeuralFoil at model_size="xxxlarge", PLUS a drag
penalty we wrote ourselves:

    S    = (m / 6) / (t / 12)^2            severity, geometry only
    dCd  = 2.5e-3 * max(0, S - 1)^2        added to the referee's Cd

That penalty is NOT real aerodynamics. We invented the formula and we
chose its size so the effect would be visible in one afternoon. We do
not have a high-fidelity solver in this room, so we mocked one.

What is NOT invented is where we put it. NeuralFoil publishes
`analysis_confidence`, and at "xxxlarge" that confidence falls from 0.97
to 0.47 in exactly this corner -- thin sections with strong camber. The
model told us it did not know. We amplified the consequence.

Three things to take away:

  1. The best-looking design your surrogate could find sat inside the
     region its own physics model understands worst. That is not bad
     luck. Optimisers are attracted to model error, because model error
     looks like performance.

  2. Your envelope did not catch it and could not have. An envelope
     tells you when you have left your data. It says nothing about your
     data being wrong.

  3. The warning was in the file you generated on Monday. Every campaign
     has a `conf` column. At "medium" the signal is weak -- 0.93 at high
     camber against 0.96 at low -- which is precisely the problem: the
     cheap model does not know that it does not know.

The fix is not a better surrogate. It is spending a few expensive
evaluations on the design you are about to commit to.
------------------------------------------------------------------------"""


def print_disclosure() -> None:
    """Call this in the debrief. It is not optional."""
    print(DISCLOSURE)


def _self_test() -> None:
    print(f"referee size: {pick_referee_size(verbose=False)}")

    X = np.array([[6.0, 0.40, 8.0, 5.0],
                  [6.0, 0.40, 12.0, 5.0],
                  [0.0, 0.40, 8.0, 5.0],
                  [3.0, 0.40, 15.0, 5.0]])
    S, d = severity(X), delta_cd(X)
    for x, s, dd in zip(X, S, d):
        print(f"  m={x[0]:4.1f} t={x[2]:4.1f}  S={s:5.2f}  dCd={dd:.5f}")
    assert d[1] == 0.0, "reference section must sit on the onset"
    assert d[2] == 0.0, "uncambered sections must be untouched"
    assert d[0] > d[3] > 0.0 or d[3] == 0.0

    ref = Referee(verbose=False)
    plain = bip.NeuralFoilBudget(budget=10 ** 6, model_size=ref.model_size,
                                 tag="plain")
    _, Y0, _ = bip.generate_dataset(plain, X, verbose=False)
    Y1, _ = ref(X)
    print("  L/D  plain -> boosted:", " ".join(
        f"{a:.0f}->{b:.0f}" for a, b in
        zip(Y0[:, 0] / Y0[:, 1], Y1[:, 0] / Y1[:, 1])))
    assert np.all(Y1[:, 1] >= Y0[:, 1] - 1e-12), "booster must never reduce drag"
    print("booster is one-signed and correctly placed: True")


if __name__ == "__main__":
    _self_test()
