"""Phase 3 dashboard: wobble animation + sliders + Hertz p_max
+ subsurface von Mises field (full-disc and contact-zoom panels).

Lifecycle: ONE FuncAnimation; slider events only set a pending flag."""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path as MplPath

from geometry.profile import theory_profile, actual_profile
from analysis.solve_frame import solve_frame
from analysis.subsurface import (build_template, superpose_field,
                                 contacts_from_frame)

MM = 1000.0
FULL_N, FULL_STRIDE = 240, 3      # full-disc grid / field frame stride
ZOOM_NX, ZOOM_NZ, ZOOM_STRIDE = 260, 180, 2


class Dashboard:
    def __init__(self, p, n_frames=72, show_diagnostics=True, lazy=False, interval=50):
        self.p, self.n_frames = p, n_frames
        self.show_diag = show_diagnostics
        self.lazy = lazy
        self.thetas = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        self._pending_rebuild = False
        self.tmpl = build_template()

        plt.rcParams["figure.figsize"] = (13.5, 8.4)
        self.fig = plt.figure()
        self.ax_geo = self.fig.add_axes([0.035, 0.50, 0.30, 0.44])
        self.ax_plot = self.fig.add_axes([0.37, 0.50, 0.25, 0.44])
        self.ax_full = self.fig.add_axes([0.035, 0.08, 0.30, 0.34])
        self.ax_zoom = self.fig.add_axes([0.37, 0.08, 0.25, 0.34])
        self._make_sliders()
        self._init_heatmaps()
        self._rebuild()
        self.anim = FuncAnimation(self.fig, self._update, frames=self.n_frames,
                                  interval=interval, blit=False, cache_frame_data=False)

    # ---------------- sliders ----------------
    def _make_sliders(self):
        specs = [("Z_p", "Z_p [-]", 9, 41, 1, int(self.p.Z_p)),
                 ("r_rp", "r_rp [mm]", 1.0, 8.0, 0.1, self.p.r_rp * MM),
                 ("R_p", "R_p [mm]", 25.0, 90.0, 1.0, self.p.R_p * MM),
                 ("e", "e [mm]", 0.2, 2.4, 0.05, self.p.e * MM),
                 ("T_in", "T_in [Nm]", 0.5, 10.0, 0.1, self.p.T_in)]
        self.sliders = {}
        ax_y = 0.44
        for key, label, lo, hi, step, val in specs:
            sax = self.fig.add_axes([0.68, ax_y, 0.27, 0.025])
            s = Slider(sax, label, lo, hi, valinit=val, valstep=step)
            s.on_changed(self._on_slider)
            self.sliders[key] = s
            ax_y -= 0.055

    def _on_slider(self, _):
        self._pending_rebuild = True

    def _apply_sliders(self):
        p = self.p
        p.Z_p = int(self.sliders["Z_p"].val)
        p.r_rp = self.sliders["r_rp"].val / MM
        p.R_p = self.sliders["R_p"].val / MM
        p.e = self.sliders["e"].val / MM
        p.T_in = self.sliders["T_in"].val

    # ---------------- rebuild (analysis) ----------------
    def _rebuild(self):
        self._apply_sliders()
        p = self.p
        t0 = time.time()
        self.results = [solve_frame(p, th) for th in self.thetas]
        self.pmax_matrix = np.array([r["p_max"] for r in self.results]).T
        self.pmax_arr = self.pmax_matrix.max(axis=0)
        self._draw_static()
        self._compute_fields()
        print(f"  [.. ] rebuilt {self.n_frames} frames + stress fields "
              f"in {time.time()-t0:.1f} s")

    def _compute_fields(self):
        if self.lazy: # nothing up front; fields computed on first use and cached
            self.full_fields, self.zoom_fields = {}, {}
            return
        p = self.p
        self.full_fields, self.zoom_fields = {}, {}
        R_m = 1.06 * (p.R_p + p.r_rp + 2 * p.e)
        xs = np.linspace(-R_m, R_m, FULL_N)
        X, Y = np.meshgrid(xs, xs)
        Pfull = np.column_stack([X.ravel(), Y.ravel()])
        for k in range(0, self.n_frames, FULL_STRIDE):
            r = self.results[k]
            vm, _ = superpose_field(Pfull, contacts_from_frame(r), self.tmpl, p.nu)
            vm = vm.reshape(FULL_N, FULL_N)
            inside = MplPath(np.column_stack([r["wx"], r["wy"]])
                             ).contains_points(Pfull).reshape(FULL_N, FULL_N)
            self.full_fields[k] = np.where(inside, vm, np.nan)
        for k in range(0, self.n_frames, ZOOM_STRIDE):
            self.zoom_fields[k] = self._zoom_field(k)

    def _zoom_field(self, k):
        p = self.p
        r = self.results[k]
        i = int(np.argmax(r["p_max"]))
        c = contacts_from_frame(r, [i])[0]
        W = max(20.0 * c["a"], 0.6e-3)            # window width [m]
        xl1 = np.linspace(-W / 2, W / 2, ZOOM_NX)
        zl1 = np.linspace(-0.15 * W, 0.85 * W, ZOOM_NZ)
        XL, ZL = np.meshgrid(xl1, zl1)
        P = (c["cp"][None, :] + XL.ravel()[:, None] * c["t"][None, :]
             + ZL.ravel()[:, None] * c["d"][None, :])
        vm, _ = superpose_field(P, contacts_from_frame(r), self.tmpl, p.nu)
        # mask above the actual (curved) surface
        _, px, py = actual_profile(p.R_p, p.e, p.Z_p, p.r_rp, p.n_profile)
        wx, wy = r["wx"], r["wy"]
        rel = np.column_stack([wx - c["cp"][0], wy - c["cp"][1]])
        surf_x = rel @ c["t"]; surf_z = rel @ c["d"]
        m = np.abs(surf_x) < W / 2
        z_surf = np.interp(np.clip(XL[0], surf_x[m].min(), surf_x[m].max()),
                           surf_x[m], surf_z[m])
        vm = vm.reshape(ZL.shape)
        vm[ZL > z_surf[None, :] + 2e-6] = np.nan
        return {"vm": vm, "xl1": xl1, "zl1": zl1, "W": W,
                "a": c["a"], 
                "hot": (c["cp"][0], c["cp"][1]),
                "pin_local": (0.0, -p.r_rp), "p_max_hot": c["p_max"],
                "profile_local": (surf_x[m], surf_z[m])}

    def _get_full_field(self, k):
        kf = min(range(0, self.n_frames, FULL_STRIDE),
                 key=lambda q: abs(q - k))
        if kf not in self.full_fields:
            r, p = self.results[kf], self.p
            R_m = 1.06 * (p.R_p + p.r_rp + 2 * p.e)
            xs = np.linspace(-R_m, R_m, FULL_N)
            X, Y = np.meshgrid(xs, xs)
            P = np.column_stack([X.ravel(), Y.ravel()])
            inside = MplPath(np.column_stack([r["wx"], r["wy"]])
                             ).contains_points(P)
            cons = [c for c in contacts_from_frame(r)
                    if c["p_max"] > 0.02 * r["p_max"].max()]  # loaded pins only
            vm, _ = superpose_field(P[inside], cons, self.tmpl, p.nu)
            out = np.full(P.shape[0], np.nan)
            out[inside] = vm
            self.full_fields[kf] = out.reshape(FULL_N, FULL_N)
        return kf

    def _get_zoom_field(self, k):
        if k not in self.zoom_fields:
            self.zoom_fields[k] = self._zoom_field(k)
        return self.zoom_fields[k]

    # ---------------- static artists ----------------
    def _draw_static(self):
        p = self.p
        for ax in (self.ax_geo, self.ax_plot): ax.clear()
        if self.show_diag:
            _, tx, ty = theory_profile(p.R_p, p.e, p.Z_p, p.n_profile)
            self.ax_geo.plot(tx * MM, ty * MM, "--", color="0.7", lw=0.7,
                             zorder=1, label="theory locus [diagnostic]")
        self.disc_poly = Polygon([[0, 0]], closed=True, facecolor="#aec7e8",
                                 alpha=0.35, edgecolor="none", zorder=2)
        self.ax_geo.add_patch(self.disc_poly)
        self.disc_line, = self.ax_geo.plot([], [], color="#1f77b4", lw=1.6,
                                           zorder=4, label="disc profile [KEY]")
        self.pin_patches = []
        hx, hy = self.results[0]["hx"], self.results[0]["hy"]
        for x, y in zip(hx, hy):
            pc = Circle((x * MM, y * MM), p.r_rp * MM, facecolor="0.85",
                        edgecolor="k", lw=0.5, zorder=5)
            self.ax_geo.add_patch(pc)
            self.pin_patches.append(pc)
        self.cursor_pt, = self.ax_geo.plot([], [], "r+", ms=12, zorder=7)
        self.ax_geo.set_aspect("equal")
        self.ax_geo.set_xlabel("x [mm]"); self.ax_geo.set_ylabel("y [mm]")
        self.ax_geo.legend(loc="upper right", fontsize=7)

        self.ax_plot.plot(np.degrees(self.thetas), self.pmax_arr / 1e6,
                          color="#1f77b4", lw=1.5)
        self.ax_plot.set_xlabel("input angle [deg]")
        self.ax_plot.set_ylabel("p_max [MPa]")
        self.ax_plot.grid(alpha=0.3)
        self.vline = self.ax_plot.axvline(0, color="r", lw=1)

    # ---------------- animation ----------------
    def _update(self, k):
        
        if self._pending_rebuild:
            self._pending_rebuild = False
            self._rebuild()
        k = k % self.n_frames
        r = self.results[k]
        p = self.p
        self.disc_line.set_data(r["wx"] * MM, r["wy"] * MM)
        self.disc_poly.set_xy(np.column_stack([r["wx"] * MM, r["wy"] * MM]))
        pmaxes = self.pmax_matrix[:, k]
        hot = pmaxes[pmaxes > 0]
        ref = hot.max() if hot.size else 1.0
        for i, pc in enumerate(self.pin_patches):
            pc.set_facecolor((1.0 - 0.6 * (pmaxes[i] / ref),
                              0.85 - 0.75 * (pmaxes[i] / ref), 0.1)
                             if pmaxes[i] > 0 else "0.85")
        self.cursor_pt.set_data([r["cx"] * MM], [r["cy"] * MM])
        self.vline.set_xdata([np.degrees(self.thetas[k])] * 2)

        # full-disc heatmap
        kf = self._get_full_field(k)
        R_m = 1.06 * (p.R_p + p.r_rp + 2 * p.e)
        vmax = 0.4 * pmaxes.max() / 1e6 if pmaxes.max() > 0 else 1.0
        self.im_full.set_data(np.ma.masked_invalid(self.full_fields[kf] / 1e6))
        self.im_full.set_extent((-R_m * MM, R_m * MM, -R_m * MM, R_m * MM))
        self.im_full.set_clim(0.0, vmax)
        self.ax_full.set_title("sigma_vm, full disc [KEY]", fontsize=9)
        self.ax_full.set_xlabel("x [mm]"); self.ax_full.set_ylabel("y [mm]")

        # zoom heatmap
        zf = self._get_zoom_field(k)
        zvm = np.ma.masked_invalid(zf["vm"] / 1e6)
        self.im_zoom.set_data(zvm)
        self.im_zoom.set_extent((zf["xl1"][0] * MM, zf["xl1"][-1] * MM,
                                 zf["zl1"][-1] * MM, zf["zl1"][0] * MM))
        self.im_zoom.set_clim(0.0, 0.4 * zf["p_max_hot"] / 1e6)
        self.zoom_curve.set_data(zf["profile_local"][0] * MM,
                                 zf["profile_local"][1] * MM)
        self.zoom_pin.set_data([zf["pin_local"][0] * MM],
                               [zf["pin_local"][1] * MM])
        self.zoom_mark.set_data([0], [0])
        self.ax_zoom.set_xlabel("tangential offset [mm]")
        self.ax_zoom.set_ylabel("depth [mm]")
        self.ax_zoom.set_title(
            f"sigma_vm @ hottest pin (p_max = {zf['p_max_hot']/1e6:.1f} MPa) [KEY]",
            fontsize=9)

        self.ax_geo.set_title(
            f"theta = {np.degrees(self.thetas[k]):5.1f} deg | i = {p.Z_c}:1 | "
            f"p_max = {r['p_max'].max()/1e6:6.1f} MPa | "
            f"F_max = {r['F_max']:6.1f} N (lit {r['F_lit']:6.1f} N)")
        self.fig.canvas.draw_idle()
        return []
        
    def _init_heatmaps(self):
        """Create heatmap artists + colorbars ONCE; rebuilds only update
        data/extents, so the axes layout never changes."""
        self.im_full = self.ax_full.imshow(
            np.full((FULL_N, FULL_N), np.nan), extent=(-1, 1, -1, 1),
            origin="lower", cmap="turbo", aspect="equal", vmin=0.0)
        self.cb_full = self.fig.colorbar(self.im_full, ax=self.ax_full,
                                        fraction=0.046)
        self.cb_full.set_label("sigma_vm [MPa]")

        self.im_zoom = self.ax_zoom.imshow(
            np.full((ZOOM_NZ, ZOOM_NX), np.nan), extent=(-1, 1, -1, 1),
            origin="upper", cmap="turbo", aspect="equal", vmin=0.0)
        self.cb_zoom = self.fig.colorbar(self.im_zoom, ax=self.ax_zoom,
                                        fraction=0.046)
        self.cb_zoom.set_label("sigma_vm [MPa]")

        self.zoom_curve, = self.ax_zoom.plot([], [], "w-", lw=1.2)
        self.zoom_pin, = self.ax_zoom.plot([], [], "o", mec="k", mfc="0.8", ms=9)
        self.zoom_mark, = self.ax_zoom.plot([], [], "w+", ms=10, mew=2)
        
    def show(self):
        plt.show()