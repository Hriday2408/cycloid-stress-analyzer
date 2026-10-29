"""Contracted (curtate) cycloid disc profile.

Theory profile : locus of pin centres expressed in the disc frame
                 (hypotrochoid, rolling circle radius R_p/Z_p, point offset e).
Actual profile : theory profile offset INWARD by roller radius r_rp
                 (the disc working surface, wraps the pins from inside).

SI units (m, rad).
"""
import numpy as np


def theory_profile(R_p: float, e: float, Z_p: int, n: int = 4000):
    """Theory (pin-locus) profile, psi in [0, 2pi)."""
    psi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = R_p * np.cos(psi) - e * np.cos(Z_p * psi)
    y = R_p * np.sin(psi) - e * np.sin(Z_p * psi)
    return psi, x, y


def derivatives(R_p, e, Z_p, psi):
    """1st/2nd derivatives d/dpsi of the theory profile (analytic)."""
    dx  = -R_p * np.sin(psi) + e * Z_p * np.sin(Z_p * psi)
    dy  =  R_p * np.cos(psi) - e * Z_p * np.cos(Z_p * psi)
    ddx = -R_p * np.cos(psi) + e * Z_p**2 * np.cos(Z_p * psi)
    ddy = -R_p * np.sin(psi) + e * Z_p**2 * np.sin(Z_p * psi)
    return dx, dy, ddx, ddy


def curvature(R_p, e, Z_p, psi):
    """Signed curvature [1/m]. kappa > 0: convex (lobe tip side),
    kappa < 0: concave (pin socket side)."""
    dx, dy, ddx, ddy = derivatives(R_p, e, Z_p, psi)
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5


def actual_profile(R_p, e, Z_p, r_rp, n: int = 4000):
    """Working profile = theory offset inward by r_rp along outward normal."""
    psi, x, y = theory_profile(R_p, e, Z_p, n)
    dx, dy, _, _ = derivatives(R_p, e, Z_p, psi)
    ds = np.hypot(dx, dy)
    nx, ny = dy / ds, -dx / ds            # outward unit normal (CCW curve)
    return psi, x - r_rp * nx, y - r_rp * ny