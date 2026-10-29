"""Per-pin contact geometry: nearest profile point, contact point,
contact normal, load angle psi, moment arm about disc centre."""
import numpy as np

from geometry.kinematics import point_polyline_distance


def _nearest_segments(qx, qy, x, y):
    """For each query point, index of nearest segment and the projected
    point on that segment."""
    ax, ay = x, y
    bx, by = np.roll(x, -1), np.roll(y, -1)
    ex, ey = bx - ax, by - ay
    seg2 = ex**2 + ey**2
    wx = np.asarray(qx)[:, None] - ax[None, :]
    wy = np.asarray(qy)[:, None] - ay[None, :]
    t = np.clip((wx * ex[None, :] + wy * ey[None, :]) / seg2[None, :], 0.0, 1.0)
    dx = wx - t * ex[None, :]
    dy = wy - t * ey[None, :]
    d2 = dx**2 + dy**2
    idx = np.argmin(d2, axis=1)
    rows = np.arange(len(idx))
    cp_x = ax[idx] + t[rows, idx] * ex[idx]
    cp_y = ay[idx] + t[rows, idx] * ey[idx]
    return idx, cp_x, cp_y


def pin_contact_geometry(hx, hy, wx, wy, cx, cy):
    """For each pin: contact point, unit normal (surface -> pin centre),
    load angle psi (vs eccentricity direction), moment arm h about disc
    centre for a force acting along -n (pin pushes disc away from pin).

    Returns dict of arrays, length Z_p."""
    dist = point_polyline_distance(hx, hy, wx, wy)
    _, cpx, cpy = _nearest_segments(hx, hy, wx, wy)

    nx, ny = hx - cpx, hy - cpy                 # surface -> pin
    n_len = np.hypot(nx, ny)
    nx, ny = nx / n_len, ny / n_len

    # direction of eccentricity (load line) in housing frame
    e_len = np.hypot(cx, cy)
    ux, uy = cx / e_len, cy / e_len

    # pin radial direction (unit, from housing centre)
    r_len = np.hypot(hx, hy)
    rx, ry = hx / r_len, hy / r_len
    psi = np.arctan2(ux * ry - uy * rx, ux * rx + uy * ry)  # sin in [_, +]

    # moment arm about DISC centre for force F*(-nx,-ny) applied at contact pt
    mx = cpx - cx
    my = cpy - cy
    h_arm = np.abs(mx * (-ny) - my * (-nx))     # |r x F_hat|

    return {"dist": dist, "cp_x": cpx, "cp_y": cpy,
            "nx": nx, "ny": ny, "psi": psi, "h": h_arm}