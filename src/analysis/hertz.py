"""Hertz line contact, pin (r_rp) vs disc actual profile (local curvature
from the exact offset-curve transformation). SI: N, m, Pa."""
import numpy as np


def e_star(E, nu):
    """Effective modulus, identical materials:
    1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2."""
    return E / (2.0 * (1.0 - nu**2))


def local_actual_curvature(k_t, r_rp):
    """Curvature of profile offset inward by r_rp from theory profile:
    k_a = k_t / (1 - r_rp*k_t). k_t signed, convex > 0 [1/m]."""
    denom = 1.0 - r_rp * k_t
    denom = np.where(denom <= 1e-9, np.nan, denom)   # undercut guard
    return k_t / denom


def line_contact(F, b, r_rp, k_actual, E_star_):
    """Returns (p_max [Pa], half-width a [m], R_equiv [m], conformal flag).
    F = 0 or conformal pocket (1/R' <= 0) -> zero stress."""
    if F <= 0.0:
        return 0.0, 0.0, np.inf, False
    inv_R = 1.0 / r_rp + k_actual          # + k: convex lobe (external)
    if inv_R <= 0.0:                        # pocket fully conformal to pin
        return 0.0, 0.0, np.inf, True
    R = 1.0 / inv_R
    Fp = F / b
    p_max = np.sqrt(Fp * E_star_ / (2.0 * np.pi * R))
    a = np.sqrt(4.0 * Fp * R / (np.pi * E_star_))
    return float(p_max), float(a), float(R), False