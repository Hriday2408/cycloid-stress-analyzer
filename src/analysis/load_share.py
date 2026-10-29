"""Force distribution: F_i = F_max * sin(psi_i) on the trailing half,
F_max solved from moment equilibrium about the disc centre.

Cross-check: classic closed form F_max = 4*T_disc/(K1*Z_p*R_p),
with short-amplitude coefficient K1 = e*Z_p/R_p.
NOTE: do NOT use 4T/(e*Z_p) here -- that form belongs to the output
pin-hole coupling, not the housing pins (overestimates by ~Z_p)."""
import numpy as np


def load_share(T_disc, psi, h, e, Z_p, R_p):
    """T_disc : torque carried by ONE disc [Nm] (= T_in / n_discs)
    Returns (F [N], F_max, F_max_literature, K1, F_bearing)."""
    engaged = np.sin(psi) > 0.0
    F = np.zeros_like(psi)
    denom = np.sum(np.sin(psi[engaged]) * h[engaged])
    F_max = T_disc / denom if denom > 0 else 0.0
    F[engaged] = F_max * np.sin(psi[engaged])
    K1 = e * Z_p / R_p
    F_lit = 4.0 * T_disc / (K1 * Z_p * R_p)
    F_bearing = T_disc / (e * (Z_p - 1))   # eccentric bearing reaction, per disc
    return F, float(F_max), float(F_lit), float(K1), float(F_bearing)