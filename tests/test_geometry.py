import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from params import CycloidParams
from geometry.profile import theory_profile, actual_profile
from geometry.checks import check_curtate, check_undercut
from geometry.kinematics import (disc_pose, pin_centers, place_profile,
                                 pin_clearances)

TOL = 5e-6  # 5 um discretization band


def test_reduction_ratio():
    p = CycloidParams()
    assert p.Z_c == p.Z_p - 1 and p.i_ratio == 20


def test_profile_closure():
    _, x0, y0 = theory_profile(0.05, 0.0015, 21, n=100)
    _, x1, y1 = theory_profile(0.05, 0.0015, 21, n=100)
    # evaluating at 0 and 2pi must coincide
    x2 = 0.05 * np.cos(2 * np.pi) - 0.0015 * np.cos(21 * 2 * np.pi)
    y2 = 0.05 * np.sin(2 * np.pi) - 0.0015 * np.sin(21 * 2 * np.pi)
    assert abs(x0[0] - x2) < 1e-12 and abs(y0[0] - y2) < 1e-12


def test_curtate_default_passes():
    p = CycloidParams()
    assert check_curtate(p.R_p, p.e, p.Z_p)["ok"]


def test_undercut_default_passes_but_huge_roller_fails():
    p = CycloidParams()
    assert check_undercut(p.R_p, p.e, p.Z_p, p.r_rp)["ok"]
    assert not check_undercut(p.R_p, p.e, p.Z_p, 0.020)["ok"]


def test_no_interference_over_full_rotation():
    """Profile is the true envelope -> no pin may penetrate the disc
    at ANY input angle (beyond numerical noise)."""
    p = CycloidParams()
    _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    hx, hy = pin_centers(p.R_p, p.Z_p)
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        cx, cy, phi = disc_pose(theta, p.e, p.Z_c)
        wx, wy = place_profile(px, py, cx, cy, phi)
        assert pin_clearances(hx, hy, wx, wy, p.r_rp).min() >= -TOL