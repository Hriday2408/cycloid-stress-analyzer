"""Subsurface stress fields for frictionless Hertzian LINE contact.

Near field : universal normalized template (Flamant kernel integrated
             against p(xi) = p_max*sqrt(1-(xi/a)^2)) in units of (x/a, z/a).
             The z = 0 row is set ANALYTICALLY (sigma_tt = sigma_dd = -p(x)),
             because point-quadrature cannot resolve depths z <~ 6/ngauss:
             there the Hertz kernel is a near-delta and single Gauss nodes
             dump w*p/z into the sum (fake vm spikes up to ~4*p_max).
Far field  : exact Flamant with resultant W = pi*a*p_max/2 for r > REACH.
All loaded contacts superposed in the global frame; von Mises with
plane-strain through-thickness stress sigma_tt = nu*(sigma_xx + sigma_yy).

Template anchors (verified against closed forms, nu = 0.3, plane strain):
  peak von Mises   0.5575 p_max at (x ~ 0, z ~ 0.7 a)   <- design number
  max Tresca shear 0.300  p_max at z = 0.786 a
  surface vm       0.4    p_max (contact center)
  far field        sigma_zz ~ -1/z (in units of p_max*a)
"""
import numpy as np

REACH = 6.0   # splice radius in units of contact half-width a


def flamant(x, z, W):
    """Stress from a normal line load W at origin (depth z >= 0).
    Returns (sxx, szz, sxz), compression negative."""
    r2 = x * x + z * z
    r2 = np.where(r2 < 1e-24, 1e-24, r2)
    f = 2.0 * W / (np.pi * r2 * r2)
    return -f * x * x * z, -f * z**3, -f * x * z * z


def build_template(nx=281, nz=181, reach=REACH, ngauss=200, nu=0.3):
    """Universal normalized near-field template (a = 1, p_max = 1).

    Depth grid starts at z = 0 with an analytic surface row; the quadrature
    rows are valid for z >~ 6/ngauss, which is auto-enforced by bumping
    ngauss if the requested dz is finer than that.
    """
    xa = np.linspace(-reach, reach, nx)
    za = np.linspace(0.0, reach, nz)
    dz = za[1] - za[0]
    ngauss = max(ngauss, int(np.ceil(6.0 / dz)) + 1)
    X, Z = np.meshgrid(xa, za)                     # (nz, nx)
    g, w = np.polynomial.legendre.leggauss(ngauss)  # nodes on [-1,1] = [-a,a]
    dW = w * np.sqrt(1.0 - g**2)                    # gauss weights x p(xi)

    M = X.size
    sTT = np.zeros(M); sDD = np.zeros(M); sTD = np.zeros(M)
    xf = X.ravel(); zf = Z.ravel()
    for j in range(ngauss):
        ax, az, axz = flamant(xf - g[j], zf, dW[j])
        sTT += ax; sDD += az; sTD += axz

    sTT = sTT.reshape(Z.shape)
    sDD = sDD.reshape(Z.shape)
    sTD = sTD.reshape(Z.shape)
    # analytic surface row (z = 0): inside the contact sigma_tt = sigma_dd
    # = -p(x), traction-free outside; quadrature cannot produce this row.
    p_surf = np.sqrt(np.maximum(0.0, 1.0 - xa**2))
    sTT[0, :] = -p_surf
    sDD[0, :] = -p_surf
    sTD[0, :] = 0.0

    # normalized von Mises (for validation/reporting)
    soo = nu * (sTT + sDD)
    vm = np.sqrt(0.5 * ((sTT - soo)**2 + (soo - sDD)**2 + (sDD - sTT)**2)
                 + 3.0 * sTD**2)
    return {"xa": xa, "za": za, "Xa": X, "Za": Z,
            "s_tt": sTT, "s_dd": sDD, "s_td": sTD, "vm": vm,
            "reach": reach, "z_min": 0.0,
            "dx": xa[1] - xa[0], "dz": dz}


def _bilinear(t, comp, xn, zn):
    xn = np.clip(xn, -t["reach"], t["reach"])
    zn = np.clip(zn, t["z_min"], t["reach"])
    fx = (xn + t["reach"]) / t["dx"]               # node-based fractional index
    fz = (zn - t["z_min"]) / t["dz"]
    j0 = np.clip(np.floor(fx).astype(int), 0, comp.shape[1] - 2)
    i0 = np.clip(np.floor(fz).astype(int), 0, comp.shape[0] - 2)
    tx = np.clip(fx - j0, 0.0, 1.0); tz = np.clip(fz - i0, 0.0, 1.0)
    c = comp
    return (c[i0, j0] * (1 - tx) * (1 - tz) + c[i0, j0 + 1] * tx * (1 - tz)
            + c[i0 + 1, j0] * (1 - tx) * tz + c[i0 + 1, j0 + 1] * tx * tz)


def _rotate(tt, dd, td, t_hat, d_hat):
    """Local (tangential, normal) stress -> global (xx, yy, xy)."""
    tx, ty = t_hat; dx, dy = d_hat
    sxx = tx * tx * tt + 2 * tx * dx * td + dx * dx * dd
    syy = ty * ty * tt + 2 * ty * dy * td + dy * dy * dd
    sxy = tx * ty * tt + (tx * dy + dx * ty) * td + dx * dy * dd
    return sxx, syy, sxy


def superpose_field(P, contacts, tmpl, nu):
    """P : (M,2) global points [m]. contacts: list of dicts with keys
    cp (2,), t (2,), d (2,) inward normal, a [m], p_max [Pa].
    Points above a contact's local surface plane (negative depth) get no
    contribution from it.
    Returns vm (M,) and (sxx, syy, sxy)."""
    P = np.asarray(P, float)
    Sxx = np.zeros(len(P)); Syy = np.zeros(len(P)); Sxy = np.zeros(len(P))
    for c in contacts:
        if c["p_max"] <= 0.0 or c["a"] <= 0.0:
            continue
        rel = P - c["cp"]
        xl = rel @ c["t"]; zl = rel @ c["d"]
        valid = zl >= 0.0                                    # inside material
        rn = np.hypot(xl, zl) / c["a"]
        near = valid & (rn < tmpl["reach"])
        if np.any(near):
            tt = _bilinear(tmpl, tmpl["s_tt"], xl[near] / c["a"], zl[near] / c["a"]) * c["p_max"]
            dd = _bilinear(tmpl, tmpl["s_dd"], xl[near] / c["a"], zl[near] / c["a"]) * c["p_max"]
            td = _bilinear(tmpl, tmpl["s_td"], xl[near] / c["a"], zl[near] / c["a"]) * c["p_max"]
            sxx, syy, sxy = _rotate(tt, dd, td, c["t"], c["d"])
            Sxx[near] += sxx; Syy[near] += syy; Sxy[near] += sxy
        far = valid & ~near
        if np.any(far):
            W = 0.5 * np.pi * c["a"] * c["p_max"]
            fx, fz, fxz = flamant(xl[far], zl[far], W)
            sxx, syy, sxy = _rotate(fx, fz, fxz, c["t"], c["d"])
            Sxx[far] += sxx; Syy[far] += syy; Sxy[far] += sxy
    stt = nu * (Sxx + Syy)
    vm = np.sqrt(0.5 * ((Sxx - stt)**2 + (stt - Syy)**2 + (Syy - Sxx)**2)
                 + 3.0 * Sxy**2)
    return vm, (Sxx, Syy, Sxy)


def contacts_from_frame(r, idx=None):
    """Build contact dicts from a solve_frame result. idx: pin subset.
    The inward normal is derived from pin center -> contact point, which is
    unambiguous: the roller always presses on the disc from outside."""
    Zp = len(r["p_max"])
    idx = range(Zp) if idx is None else idx
    out = []
    for i in idx:
        if r["p_max"][i] > 0.0 and r["a"][i] > 0.0:
            cp = np.array([r["cp_x"][i], r["cp_y"][i]])
            v = cp - np.array([r["hx"][i], r["hy"][i]])   # pin -> contact
            d = v / np.linalg.norm(v)                     # = into the disc
            out.append({"cp": cp, "d": d, "t": np.array([-d[1], d[0]]),
                        "a": r["a"][i], "p_max": r["p_max"][i], "i": i})
    return out