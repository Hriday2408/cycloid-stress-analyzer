"""One-stop per-frame solver: pose -> contact geometry -> forces -> Hertz.
Used by both the dashboard and the batch scans."""
import numpy as np

from geometry.profile import theory_profile, actual_profile, derivatives, curvature
from geometry.kinematics import disc_pose, pin_centers, place_profile
from analysis.contact_detect import pin_contact_geometry
from analysis.load_share import load_share
from analysis.hertz import e_star, local_actual_curvature, line_contact


def solve_frame(p, theta, _cache={}):
    """Returns dict with profile points (housing frame), per-pin F, p_max,
    a, psi, and summary scalars. Geometry results cached per (R_p,e,Z_p,r_rp,n)."""
    key = (p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    if key not in _cache:
        _, tx, ty = theory_profile(p.R_p, p.e, p.Z_p, p.n_profile)
        psi_t, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
        dx, dy, ddx, ddy = derivatives(p.R_p, p.e, p.Z_p, psi_t)
        k_t = curvature(p.R_p, p.e, p.Z_p, psi_t)
        k_a = local_actual_curvature(k_t, p.r_rp)
        hx, hy = pin_centers(p.R_p, p.Z_p)
        _cache[key] = {"psi_t": psi_t, "tx": tx, "ty": ty, "px": px, "py": py,
                       "k_a": k_a, "hx": hx, "hy": hy}
    g = _cache[key]

    cx, cy, phi = disc_pose(theta, p.e, p.Z_c)
    wx, wy = place_profile(g["px"], g["py"], cx, cy, phi)
    cg = pin_contact_geometry(g["hx"], g["hy"], wx, wy, cx, cy)

    T_disc = p.T_in / p.n_discs
    F, F_max, F_lit, K1, F_bearing = load_share(T_disc, cg["psi"], cg["h"], p.e, p.Z_p, p.R_p)

    Est = e_star(p.E, p.nu)
    seg_idx = np.zeros(p.Z_p, dtype=int)
    # map each pin to nearest profile index for local curvature
    from analysis.contact_detect import _nearest_segments
    seg_idx, _, _ = _nearest_segments(g["hx"], g["hy"], wx, wy)
    p_max = np.zeros(p.Z_p)
    a_hw = np.zeros(p.Z_p)
    for i in range(p.Z_p):
        if F[i] > 0 and np.isfinite(g["k_a"][seg_idx[i]]):
            p_max[i], a_hw[i], _, _ = line_contact(
                F[i], p.b, p.r_rp, g["k_a"][seg_idx[i]], Est)

    return {"wx": wx, "wy": wy, "cx": cx, "cy": cy, "phi": phi,
            "F": F, "F_max": F_max, "F_lit": F_lit, "K1": K1, "F_bearing": F_bearing,
            "p_max": p_max, "a": a_hw, "psi": cg["psi"],
            "hx": g["hx"], "hy": g["hy"],
            "cp_x": cg["cp_x"], "cp_y": cg["cp_y"],
            "nx": cg["nx"], "ny": cg["ny"]}