"""Phase 5: export CSV tables, Ansys field grid, DXF profile, MP4/GIF."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ROOT_OUT = ROOT / "outputs"

import time                                                   # noqa: E402
import numpy as np                                            # noqa: E402
from params import CycloidParams                              # noqa: E402
from viz.dashboard import Dashboard                           # noqa: E402
from utils.report import header, key                          # noqa: E402
from exporters import (export_contacts_cycle,                 # noqa: E402
                       export_cycle_summary, export_hot_field_csv,
                       export_profile_dxf_with_pins, save_animation)


def main():
    p = CycloidParams.from_json(ROOT / "config" / "default_params.json")
    header("PHASE 5 - EXPORT (CSV / field grid / DXF / video)")
    ROOT_OUT.mkdir(exist_ok=True)

    sim = Dashboard(p, n_frames=360, lazy=True, interval=25)
    thetas = sim.thetas
    k_hot = int(np.unravel_index(np.argmax(sim.pmax_matrix),
                                 sim.pmax_matrix.shape)[1])
    key("Hottest frame", f"theta = {np.degrees(thetas[k_hot]):.1f} deg, "
        f"p_max = {sim.pmax_matrix[:, k_hot].max()/1e6:.1f} MPa")

    t0 = time.time()
    f1 = export_contacts_cycle(
        sim.results, thetas, ROOT_OUT / "contacts_cycle.csv")
    f2 = export_cycle_summary(
        sim.results, thetas, ROOT_OUT / "pmax_cycle.csv")
    key("contacts_cycle.csv", f1.name,
        f"{(ROOT_OUT / f1.name).stat().st_size/1e3:.0f} kB")
    key("pmax_cycle.csv", f2.name,
        f"{(ROOT_OUT / f2.name).stat().st_size/1e3:.0f} kB")

    f3, ratio, depth = export_hot_field_csv(
        p, sim.results[k_hot], sim.tmpl, np.degrees(thetas[k_hot]),
        ROOT_OUT / "field_hot_frame.csv")
    key("field_hot_frame.csv", f3.name,
        f"fine-grid vm/p_max = {ratio:.3f} @ depth {depth:.2f} a "
        f"(anchor 0.557 @ 0.7a)")
    key("  -> Ansys check", "import CSV, compare sigma_vm lobe: "
        "peak value AND depth must match within a few %")

    f4 = export_profile_dxf_with_pins(
        p, sim.results, ROOT_OUT / "profile.dxf")
    if f4:
        key("profile.dxf", Path(f4).name, "mm units, pins on layer PINS")

    print("[.. ] saving animation: first pass computes all lazy fields, "
          "expect ~2-4 min ...")
    t1 = time.time()
    kind = save_animation(sim, ROOT_OUT / "phase5_animation.mp4", fps=24)
    ext = "mp4" if kind == "mp4" else "gif"
    key(f"phase5_animation.{ext}", f"saved in {time.time()-t1:.0f} s")

    key("DONE", f"5 files in {ROOT_OUT}")


if __name__ == "__main__":
    main()