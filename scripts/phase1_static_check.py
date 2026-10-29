"""PHASE 1 - static verification: design checks + one rendered frame.

Run from repo root:
    python scripts/phase1_static_check.py            (theta = 30 deg)
    python scripts/phase1_static_check.py 0          (any theta in deg)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from params import CycloidParams                      # noqa: E402
from geometry.checks import (check_curtate, check_undercut,   # noqa: E402
                             min_concave_radius)
from geometry.kinematics import (disc_pose, pin_centers,      # noqa: E402
                                 place_profile, pin_clearances)
from geometry.profile import actual_profile                   # noqa: E402
from utils.report import header, key, info, rule, ok_fail     # noqa: E402
from viz.static_view import draw_frame                        # noqa: E402

CONTACT_TOL = 10e-6      # 10 um: numerical noise band for "in contact"


def main():
    theta_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    p = CycloidParams.from_json(ROOT / "config" / "default_params.json")

    header(f"PHASE 1 - STATIC VERIFICATION | theta = {theta_deg:.1f} deg")

    # ---------------- design checks ----------------
    cur = check_curtate(p.R_p, p.e, p.Z_p)
    key("Reduction ratio i", f"{p.i_ratio} : 1", f"(Z_c = {p.Z_c}, Z_p = {p.Z_p})")
    key("Curtate condition", ok_fail(cur["ok"]),
        f"(e = {p.e*1e3:.3f} mm < R_p/Z_p = {cur['limit']*1e3:.3f} mm)")

    unc = check_undercut(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    key("Undercut (offset) check", ok_fail(unc["ok"]),
        f"(rho_min,convex = {unc['rho_min']*1e3:.2f} mm > r_rp = {p.r_rp*1e3:.2f} mm)")
    info("Min concave rho (pockets)", f"{min_concave_radius(p.R_p, p.e, p.Z_p)*1e3:.2f} mm",
         "(sets internal-contact geometry later, no offset risk)")
    info("Profile points", f"{p.n_profile}")
    rule()

    # ---------------- static frame ----------------
    theta = np.radians(theta_deg)
    _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    cx, cy, phi = disc_pose(theta, p.e, p.Z_c)
    wx, wy = place_profile(px, py, cx, cy, phi)
    hx, hy = pin_centers(p.R_p, p.Z_p)
    cl = pin_clearances(hx, hy, wx, wy, p.r_rp)

    in_contact = cl <= CONTACT_TOL
    worst = cl.min()
    key("Max interference", f"{abs(min(worst, 0))*1e6:.2f} um",
        "(target ~0; larger = undercut or bad discretization)")
    key("Pins in geometric contact", f"{int(in_contact.sum())} / {p.Z_p}",
        "(ideal profile -> ALL pins touch; force-carrying subset comes in Phase 2)")
    n_clear = int((~in_contact).sum())
    if n_clear > 0:
        info("Mean clearance (non-contact pins)", f"{cl[~in_contact].mean()*1e3:.3f} mm")
    else:
        info("Mean clearance (non-contact pins)", "n/a - all pins touch (ideal geometry)")
    info("Max clearance", f"{cl.max()*1e3:.3f} mm", "(0 for ideal unmodified profile)")
    info("Disc pose (cx, cy, phi)",
        f"({cx*1e3:.3f}, {cy*1e3:.3f}, {np.degrees(phi):.3f} deg) mm/deg")
    rule()

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_frame(ax, p, theta_deg, contact_tol=CONTACT_TOL)
    ax.legend(loc="upper right", fontsize=8)
    out = ROOT / "outputs" / f"phase1_theta{theta_deg:g}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [KEY] Frame saved -> {out}")
    plt.show()

    if worst < -CONTACT_TOL:
        print("\n  !! [KEY] INTERFERENCE DETECTED beyond numerical noise - "
              "check undercut / parameters.")


if __name__ == "__main__":
    main()