"""Phase 5: exporters - CSV tables, Ansys-ready subsurface field,
DXF profile, animation video. All SI units unless noted."""
import csv
import numpy as np

from geometry.profile import actual_profile
from analysis.subsurface import contacts_from_frame, superpose_field

MM = 1000.0
UM = 1e6


# ---------------------------------------------------------------- CSV 1
def export_contacts_cycle(results, thetas, path, b=None):
    """Long table: one row per (frame, pin). F_prime = F / roller width."""
    if b is None:
        b = 0.010                       # default roller width 10 mm [m]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# cycloid drive contacts over one input revolution"])
        w.writerow([f"# F_prime assumes roller width b = {b*MM:.1f} mm"])
        w.writerow(["theta_deg", "pin", "psi_deg", "F_N",
                    "F_prime_N_per_m", "p_max_MPa", "a_um", "loaded"])
        for k, r in enumerate(results):
            th = np.degrees(thetas[k])
            for i in range(len(r["F"])):
                F = float(r["F"][i])
                psi = r["psi"][i] if r.get("psi") is not None else np.nan
                w.writerow([
                    f"{th:.3f}", i,
                    "" if not np.isfinite(psi) else f"{np.degrees(psi):.3f}",
                    f"{F:.4f}",
                    f"{F / b:.1f}",
                    f"{float(r['p_max'][i]) / 1e6:.4f}",
                    f"{float(r['a'][i]) * UM:.3f}",
                    1 if F > 0 else 0])
    return path


# ---------------------------------------------------------------- CSV 2
def export_cycle_summary(results, thetas, path):
    """One row per frame: driven quantities for plots and reports."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# per-frame summary over one input revolution"])
        w.writerow(["theta_deg", "p_max_MPa", "n_loaded",
                    "F_max_N", "F_lit_N", "F_bearing_N"])
        for k, r in enumerate(results):
            pm = np.asarray(r["p_max"])
            w.writerow([f"{np.degrees(thetas[k]):.3f}",
                        f"{pm.max() / 1e6:.4f}",
                        int((np.asarray(r["F"]) > 0).sum()),
                        f"{r['F_max']:.3f}", f"{r['F_lit']:.3f}",
                        f"{r['F_bearing']:.3f}"])
    return path


# ---------------------------------------------------------------- CSV 3
def hot_frame_field(p, r, tmpl, nx=401, nz=301,
                    w_min=0.15e-3, n_span=20.0):
    """Fine subsurface vm field around the hottest pin of one frame.
    Independent of the dashboard display grid: finer dz (~0.19 a).
    Returns local AND global coordinates."""
    i = int(np.argmax(r["p_max"]))
    c = contacts_from_frame(r, [i])[0]
    W = max(n_span * c["a"], w_min)
    xl = np.linspace(-W / 2, W / 2, nx)
    zl = np.linspace(-0.15 * W, 0.85 * W, nz)
    XL, ZL = np.meshgrid(xl, zl)
    P = (c["cp"][None, :] + XL.ravel()[:, None] * c["t"][None, :]
         + ZL.ravel()[:, None] * c["d"][None, :])
    XG = P[:, 0].reshape(ZL.shape)
    YG = P[:, 1].reshape(ZL.shape)
    cons = [q for q in contacts_from_frame(r)
            if q["p_max"] > 0.02 * r["p_max"].max()]
    vm, _ = superpose_field(P, cons, tmpl, p.nu)
    vm = vm.reshape(ZL.shape)
    # mask air above the curved surface
    wx, wy = r["wx"], r["wy"]
    rel = np.column_stack([wx - c["cp"][0], wy - c["cp"][1]])
    surf_x = rel @ c["t"]
    surf_z = rel @ c["d"]
    m = np.abs(surf_x) < W / 2
    z_surf = np.interp(np.clip(XL[0], surf_x[m].min(), surf_x[m].max()),
                       surf_x[m], surf_z[m])
    material = ZL <= z_surf[None, :] + 2e-6
    return {"xl": xl, "zl": zl, "XG": XG, "YG": YG, "vm": vm,
            "material": material, "a": c["a"], "W": W,
            "p_max_hot": c["p_max_hot"] if "p_max_hot" in c else c["p_max"],
            "pin": i}


def export_hot_field_csv(p, r, tmpl, theta_deg, path, nx=401, nz=301):
    """Ansys-comparison grid: local + global coords, material flag, vm."""
    f = hot_frame_field(p, r, tmpl, nx=nx, nz=nz)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# subsurface von Mises field @ hottest contact"])
        w.writerow([f"# theta = {theta_deg:.2f} deg, pin = {f['pin']}"])
        w.writerow([f"# R_p = {p.R_p*MM:.2f} mm, e = {p.e*MM:.3f} mm, "
                    f"Z_p = {p.Z_p}, r_rp = {p.r_rp*MM:.2f} mm, "
                    f"T_in = {p.T_in:.2f} Nm, nu = {p.nu}"])
        w.writerow([f"# p_max_hot = {f['p_max_hot']/1e6:.3f} MPa, "
                    f"a = {f['a']*UM:.3f} um"])
        w.writerow(["# coords in meters, stress in Pa; "
                    "material=1 rows carry valid sigma_vm"])
        w.writerow(["x_local_m", "z_local_m", "x_global_m", "y_global_m",
                    "material", "sigma_vm_Pa"])
        nz_, nx_ = f["vm"].shape
        for iz in range(nz_):
            for ix in range(nx_):
                mat = int(f["material"][iz, ix])
                w.writerow([
                    f"{f['xl'][ix]:.6e}", f"{f['zl'][iz]:.6e}",
                    f"{f['XG'][iz, ix]:.6e}", f"{f['YG'][iz, ix]:.6e}",
                    mat,
                    f"{f['vm'][iz, ix]:.6e}" if mat else ""])
    # [KEY] anchor check on the fine grid
    j = np.nanargmax(f["vm"])
    iz_pk, _ = np.unravel_index(j, f["vm"].shape)
    ratio = np.nanmax(f["vm"]) / f["p_max_hot"]
    depth = f["zl"][iz_pk] / f["a"]
    return path, ratio, depth


# ---------------------------------------------------------------- DXF
def export_profile_dxf(p, path):
    """Cycloid profile (mm) + pin circles. Needs ezdxf; returns path
    or None if ezdxf is missing."""
    try:
        import ezdxf
    except ImportError:
        print("[.. ] ezdxf not installed -> DXF skipped "
              "(pip install ezdxf)")
        return None
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4          # 4 = millimeters
    msp = doc.modelspace()
    _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    pts = np.column_stack([px, py]) * MM
    tuples = [(float(x), float(y)) for x, y in pts]
    try:                                 # smooth spline, fit points
        from ezdxf.math import fit_points_to_cad_cv
        cv = fit_points_to_cad_cv(tuples, degree=3)
        msp.add_open_spline(cv.control_points, degree=3,
                            knots=cv.knot_values())
    except Exception:                    # API differences -> polyline
        msp.add_lwpolyline(tuples, format="xy", close=True)
    for x, y in zip(p.__dict__.get("hx", []), p.__dict__.get("hy", [])):
        pass                             # pins come from solve_frame below
    return path


def export_profile_dxf_with_pins(p, results, path):
    """Profile + pin roller circles at theta=0 (geometry is static)."""
    try:
        import ezdxf
    except ImportError:
        print("[.. ] ezdxf not installed -> DXF skipped "
              "(pip install ezdxf)")
        return None
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
    tuples = [(float(x), float(y)) for x, y in np.column_stack([px, py]) * MM]
    try:
        from ezdxf.math import fit_points_to_cad_cv
        cv = fit_points_to_cad_cv(tuples, degree=3)
        msp.add_open_spline(cv.control_points, degree=3,
                            knots=cv.knot_values())
    except Exception:
        msp.add_lwpolyline(tuples, format="xy", close=True)
    for x, y in zip(results[0]["hx"], results[0]["hy"]):
        msp.add_circle((float(x) * MM, float(y) * MM),
                       float(p.r_rp) * MM, dxfattribs={"layer": "PINS"})
    doc.saveas(path)
    return path


# ---------------------------------------------------------------- video
def save_animation(sim, path_mp4, path_gif=None, fps=24):
    """Try MP4 via ffmpeg; fall back to GIF via Pillow. Returns writer
    name actually used. NOTE: first pass computes all lazy fields."""
    try:
        sim.anim.save(path_mp4, writer="ffmpeg", fps=fps)
        return "mp4"
    except Exception as e:
        print(f"[.. ] ffmpeg unavailable ({e.__class__.__name__}) "
              f"-> falling back to GIF")
        from matplotlib.animation import PillowWriter
        gif = path_gif or str(path_mp4).replace(".mp4", ".gif")
        sim.anim.save(gif, writer=PillowWriter(fps=max(8, fps // 2)))
        return "gif"