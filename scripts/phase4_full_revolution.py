"""Phase 4: full-revolution simulation.
360-frame continuous sweep, lazy on-demand stress fields, cycle metrics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import time                                                    # noqa: E402
import numpy as np                                             # noqa: E402
from params import CycloidParams                               # noqa: E402
from viz.dashboard import Dashboard                            # noqa: E402
from utils.report import header, key                           # noqa: E402


def cycle_metrics(sim):
    pm = sim.pmax_matrix                      # (n_pins, n_frames)
    angles = np.degrees(sim.thetas)
    i_hot, k_hot = np.unravel_index(np.argmax(pm), pm.shape)
    loaded = (pm > 0).sum(axis=0)
    key("Global peak p_max",
        f"{pm.max()/1e6:.1f} MPa @ theta = {angles[k_hot]:.1f} deg, "
        f"pin #{i_hot}")
    key("Loaded contacts per frame",
        f"min {loaded.min()} / mean {loaded.mean():.1f} / max {loaded.max()}",
        f"(expected ~{sim.p.Z_p // 2})")
    key("Load pulses per pin",
        "ONE per input revolution -> fatigue cycles = input RPM x 1")
    print("[.. ] subsurface sweep, every 10th frame "
          "(anchor: vm_peak ~ 0.557 p_max @ depth ~0.7a)")
    for k in range(0, sim.n_frames, 10):
        zf = sim._get_zoom_field(k)
        vm = zf["vm"]
        iz, _ = np.unravel_index(np.nanargmax(vm), vm.shape)
        a = zf["a"]                  # valid while W not clamped
        print(f"  [.. ] theta={np.degrees(sim.thetas[k]):5.1f} deg  "
              f"vm/p_max = {np.nanmax(vm)/zf['p_max_hot']:.3f}  "
              f"@ depth = {zf['zl1'][iz]/a:.2f} a")


def main():
    p = CycloidParams.from_json(ROOT / "config" / "default_params.json")
    header("PHASE 4 - FULL-REVOLUTION SIMULATION "
           "(360 frames, lazy stress fields)")
    key("Reduction ratio", f"{p.i_ratio} : 1")
    t0 = time.time()
    sim = Dashboard(p, n_frames=360, lazy=True, interval=25)
    key("Sweep solved", f"360 frames in {time.time()-t0:.1f} s")
    cycle_metrics(sim)
    key("Playback", "FIRST revolution stutters (lazy fields); rev 2+ smooth")
    sim.show()


if __name__ == "__main__":
    main()