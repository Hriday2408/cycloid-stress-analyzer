"""Static frame rendering. KEY layers: disc profile (solid blue),
contact pins (red). Diagnostic layers (dashed gray): theory locus,
pin circle, eccentric orbit - toggle with show_diagnostics=False."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from geometry.profile import theory_profile, actual_profile
from geometry.kinematics import disc_pose, pin_centers, place_profile, pin_clearances

MM = 1000.0


def draw_frame(ax, p, theta_deg, contact_tol=10e-6, show_diagnostics=True):
    """Draw one static frame. Returns per-pin clearances [m]."""
    theta = np.radians(theta_deg)
    _, tx, ty = theory_profile(p.R_p, p.e, p.Z_p, p.n_profile)
    _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    cx, cy, phi = disc_pose(theta, p.e, p.Z_c)
    wx, wy = place_profile(px, py, cx, cy, phi)
    hx, hy = pin_centers(p.R_p, p.Z_p)
    cl = pin_clearances(hx, hy, wx, wy, p.r_rp)

    # --- KEY layer: disc -----------------------------------------------
    ax.add_patch(Polygon(np.column_stack([wx * MM, wy * MM]), closed=True,
                         facecolor="#aec7e8", alpha=0.35, edgecolor="none",
                         zorder=2))
    ax.plot(wx * MM, wy * MM, color="#1f77b4", lw=1.8, zorder=4,
            label="disc profile [KEY]")

    # --- diagnostic layers ----------------------------------------------
    if show_diagnostics:
        twx, twy = place_profile(tx, ty, cx, cy, phi)
        ax.plot(twx * MM, twy * MM, "--", color="0.65", lw=0.8, zorder=3,
                label="theory locus [diagnostic]")
        ax.add_patch(Circle((0, 0), p.R_p * MM, fill=False, ls=":",
                            edgecolor="0.65", lw=0.8, zorder=1))
        ax.add_patch(Circle((0, 0), p.e * MM, fill=False, ls=":",
                            edgecolor="0.65", lw=0.8, zorder=1))

    # --- KEY layer: pins by contact state --------------------------------
    for x, y, c in zip(hx, hy, cl):
        if c <= contact_tol:
            fc = "tab:red"       # contact  [KEY]
        elif c < 0.5e-3:
            fc = "tab:orange"    # near-contact (transition zone)
        else:
            fc = "0.78"          # clear
        ax.add_patch(Circle((x * MM, y * MM), p.r_rp * MM, facecolor=fc,
                            edgecolor="k", lw=0.6, alpha=0.9, zorder=5))
    ax.plot(hx * MM, hy * MM, "k.", ms=3, zorder=6)
    ax.plot(cx * MM, cy * MM, "k+", ms=10, zorder=7)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Phase 1 static frame | theta = {theta_deg:.1f} deg "
                 f"| i = {p.Z_c}:1 | e = {p.e*MM:.2f} mm")
    return cl