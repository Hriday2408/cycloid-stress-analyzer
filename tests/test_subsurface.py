import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.subsurface import build_template, superpose_field, contacts_from_frame
from params import CycloidParams
from analysis.solve_frame import solve_frame


def test_template_peak_vm_bands():
    """Textbook line contact (plane strain, nu=0.3): peak von Mises
    = 0.557 p_max at depth ~0.7 a on the contact centerline.
    (0.30 is the TRESCA value - do not confuse the two.)"""
    t = build_template(nx=201, nz=141, ngauss=160)
    k = np.unravel_index(np.argmax(t["vm"]), t["vm"].shape)
    assert 0.49 <= t["vm"][k] <= 0.62          # magnitude band (von Mises!)
    assert 0.5 <= t["Za"][k] <= 1.1            # depth band  (units of a)
    assert abs(t["Xa"][k]) <= 0.35             # under the contact center


def test_template_surface_row():
    """z = 0 row is analytic: sigma_tt = sigma_dd = -p(x) inside the contact,
    giving vm = 0.4 p_max at the center (nu = 0.3)."""
    t = build_template(nx=201, nz=141, ngauss=160)
    ic = np.argmin(np.abs(t["xa"]))            # center column
    assert abs(t["s_tt"][0, ic] + 1.0) < 1e-12
    assert abs(t["s_dd"][0, ic] + 1.0) < 1e-12
    assert abs(t["vm"][0, ic] - 0.4) < 1e-12
    assert abs(t["s_tt"][0, -1]) < 1e-12       # traction-free outside contact


def test_template_lateral_symmetry():
    t = build_template(nx=201, nz=141, ngauss=160)
    assert np.allclose(t["vm"], t["vm"][:, ::-1], atol=1e-6)


def test_template_closed_form_anchors():
    """On-axis stresses must match the Smith-Liu closed forms
        sigma_dd = -1/sqrt(1+z^2)
        sigma_tt = -[(1+2z^2)/sqrt(1+z^2) - 2z]
    (guards against any quadrature regression)."""
    t = build_template(nx=201, nz=141, ngauss=160)
    ic = np.argmin(np.abs(t["xa"]))
    for row in (2, 12, 23):                    # z = 0.086, 0.514, 0.986
        z = t["za"][row]
        assert abs(t["s_dd"][row, ic] + 1.0 / np.sqrt(1 + z**2)) < 1e-6
        assert abs(t["s_tt"][row, ic]
                   + ((1 + 2 * z**2) / np.sqrt(1 + z**2) - 2 * z)) < 1e-6


def test_gauss_load_total():
    """Gauss sum must reproduce W = pi/2 (a=1, p_max=1)."""
    g, w = np.polynomial.legendre.leggauss(200)
    assert abs((w * np.sqrt(1 - g**2)).sum() - np.pi / 2) < 1e-6


def test_field_superposition_sane():
    """Resolved local patch at the most-loaded pin: peak ~0.56 p_max at
    depth ~0.7 a. (The contact field spans +-3a ~ 15 um; a grid that
    resolves it must be local to the contact.)"""
    p = CycloidParams()
    r = solve_frame(p, 0.7)
    contacts = contacts_from_frame(r)
    assert len(contacts) > 0
    c0 = max(contacts, key=lambda c: c["p_max"])
    assert abs(c0["p_max"] - r["p_max"].max()) < 1e-9      # it IS the global max
    a0 = c0["a"]
    s = np.linspace(-5 * a0, 5 * a0, 201)
    d = np.linspace(-1 * a0, 4 * a0, 201)
    S, D = np.meshgrid(s, d)
    P = c0["cp"] + S.ravel()[:, None] * c0["t"] + D.ravel()[:, None] * c0["d"]
    vm, _ = superpose_field(P, contacts, build_template(201, 141), p.nu)
    ratio = vm.max() / c0["p_max"]
    assert 0.45 <= ratio <= 0.65
    k = int(np.argmax(vm))
    assert 0.4 <= D.ravel()[k] / a0 <= 1.1                 # subsurface peak ~0.7 a


def test_coarse_disc_far_field():
    """Full-disc grid (cell ~0.8 mm) cannot resolve a ~ 5 um contact fields;
    it only sees the far-field tail -> ratio must stay tiny (< 0.08).
    This is the correct physics behind the old '0.0123' observation, and a
    regression guard against fake near-surface spikes."""
    p = CycloidParams()
    r = solve_frame(p, 0.7)
    R_m = 1.1 * (p.R_p + p.r_rp + 2 * p.e)
    xs = np.linspace(-R_m, R_m, 160)
    X, Y = np.meshgrid(xs, xs)
    P = np.column_stack([X.ravel(), Y.ravel()])
    vm, _ = superpose_field(P, contacts_from_frame(r), build_template(161, 101), p.nu)
    assert vm.max() / r["p_max"].max() <= 0.08