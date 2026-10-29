"""Design validity checks. Every check returns dict(ok, ...) for dashboard use."""
import numpy as np
from .profile import curvature


def check_curtate(R_p, e, Z_p):
    """Curtate condition: e < R_p / Z_p (else cusped/loopy profile)."""
    limit = R_p / Z_p
    return {"ok": bool(e < limit), "limit": limit}


def min_convex_radius(R_p, e, Z_p, n=4000):
    """Smallest radius of curvature among CONVEX regions (kappa > 0).

    Only convex regions shrink radius under the inward offset, so this
    is the governing number for the undercut criterion."""
    psi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    k = curvature(R_p, e, Z_p, psi)
    k_pos = k[k > 0]
    return float("inf") if k_pos.size == 0 else float(1.0 / k_pos.max())


def min_concave_radius(R_p, e, Z_p, n=4000):
    """Smallest radius of curvature among CONCAVE regions (kappa < 0).

    Diagnostic only: inward offset grows the radius there, no undercut
    risk, but it sets the local contact geometry (internal vs external)."""
    psi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    k = curvature(R_p, e, Z_p, psi)
    k_neg = k[k < 0]
    return float("inf") if k_neg.size == 0 else float(1.0 / np.abs(k_neg).max())


def check_undercut(R_p, e, Z_p, r_rp, n=4000):
    """Undercut: actual profile self-intersects if r_rp > min convex rho."""
    rho = min_convex_radius(R_p, e, Z_p, n)
    return {"ok": bool(rho > r_rp), "rho_min": rho}