"""Wobble-plate kinematics: disc pose per input angle, pin layout,
pin-to-profile clearance (contact detection groundwork for Phase 2)."""
import numpy as np


def disc_pose(theta, e, Z_c):
    """Disc centre (housing frame) and disc rotation angle.

    theta : input (eccentric) angle [rad]
    returns (cx, cy, phi) with phi = -theta/Z_c (disc counter-rotates,
    output speed = omega_in / Z_c -> reduction ratio i = Z_c)."""
    return e * np.cos(theta), e * np.sin(theta), -theta / Z_c


def pin_centers(R_p, Z_p):
    """Fixed pin centres in the housing frame."""
    a = 2.0 * np.pi * np.arange(Z_p) / Z_p
    return R_p * np.cos(a), R_p * np.sin(a)


def place_profile(px, py, cx, cy, phi):
    """Transform disc-frame profile points into the housing frame."""
    c, s = np.cos(phi), np.sin(phi)
    return cx + c * px - s * py, cy + s * px + c * py


def point_polyline_distance(qx, qy, x, y):
    """Min distance from each query point (qx,qy) to closed polyline (x,y)."""
    ax, ay = x, y
    bx, by = np.roll(x, -1), np.roll(y, -1)
    ex, ey = bx - ax, by - ay
    seg2 = ex**2 + ey**2
    wx = np.asarray(qx)[:, None] - ax[None, :]
    wy = np.asarray(qy)[:, None] - ay[None, :]
    t = np.clip((wx * ex[None, :] + wy * ey[None, :]) / seg2[None, :], 0.0, 1.0)
    dx = wx - t * ex[None, :]
    dy = wy - t * ey[None, :]
    return np.sqrt((dx**2 + dy**2).min(axis=1))


def pin_clearances(pin_x, pin_y, prof_x, prof_y, r_rp):
    """clearance_i = dist(pin_i -> profile) - r_rp.
    < 0 : interference (should only be numerical noise)
    ~ 0 : pin in contact
    > 0 : pin clear."""
    d = point_polyline_distance(pin_x, pin_y, prof_x, prof_y)
    return d - r_rp