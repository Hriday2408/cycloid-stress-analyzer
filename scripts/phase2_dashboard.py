"""PHASE 2 - interactive dashboard: sliders + animation + Hertz stress."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from params import CycloidParams            # noqa: E402
from viz.dashboard import Dashboard         # noqa: E402
from utils.report import header, key        # noqa: E402


def main():
    p = CycloidParams.from_json(ROOT / "config" / "default_params.json")
    header("PHASE 2 - DASHBOARD (contact forces + Hertz p_max)")
    key("Reduction ratio", f"{p.i_ratio} : 1")
    
    K1 = p.e * p.Z_p / p.R_p
    key("Literature F_max per pin", f"{4*(p.T_in/p.n_discs)/(K1*p.Z_p*p.R_p):.2f} N",
        "(= 4T_disc/(K1*Z_p*R_p), K1 = e*Z_p/R_p = %.2f)" % K1)
    key("Eccentric bearing force (per disc)",
        f"{(p.T_in/p.n_discs)/(p.e*p.Z_c):.1f} N", "(pin-force resultant must balance this)")
    dash = Dashboard(p, n_frames=72)
    dash.show()


if __name__ == "__main__":
    main()