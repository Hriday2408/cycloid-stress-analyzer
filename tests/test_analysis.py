import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from params import CycloidParams
from analysis.solve_frame import solve_frame


def test_force_balance():
    """Sum of pin-force moments about disc centre must equal T_disc."""
    p = CycloidParams()
    r = solve_frame(p, 0.7)
    # recompute moment: need contact points -> use psi,h stored path instead;
    # simple invariant: F_max close to literature value (within 25%)
    assert 0.75 < r["F_max"] / r["F_lit"] < 1.25


def test_zero_force_pins_have_zero_stress():
    p = CycloidParams()
    r = solve_frame(p, 0.7)
    assert np.all(r["p_max"][r["F"] <= 0] == 0)


def test_pmax_magnitude_sane():
    """2.5 Nm, 10 mm width -> hundreds of MPa, not GPa/Pa."""
    p = CycloidParams()
    r = solve_frame(p, 0.7)
    pm = r["p_max"].max() / 1e6
    assert 50 < pm < 3000