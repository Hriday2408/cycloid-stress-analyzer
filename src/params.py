"""Central parameter container. SI units throughout: m, N, Pa, Nm, rad."""
import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class CycloidParams:
    # --- geometry (independent, slider-driven later) ---
    Z_p: int = 21            # number of housing pins
    r_rp: float = 4e-3       # roller (pin) radius [m]
    R_p: float = 50e-3       # pin-centre circle radius [m]
    e: float = 1.5e-3        # eccentricity [m]
    # --- operating (independent) ---
    T_in: float = 2.5        # input torque [Nm]
    b: float = 10e-3         # disc width [m] (Hertz line contact uses this)
    n_discs: int = 2         # discs, 180 deg out of phase
    # --- material (independent, same for disc & pins for now) ---
    E: float = 208e9         # Young's modulus [Pa]
    nu: float = 0.3          # Poisson ratio [-]
    # --- numerics ---
    n_profile: int = 4000    # profile discretization points

    # ---- dependent (read-only, derived) ----
    @property
    def Z_c(self) -> int:
        """Disc lobes. Enforced relation: Z_c = Z_p - 1."""
        return self.Z_p - 1

    @property
    def i_ratio(self) -> int:
        """Reduction ratio, single stage, fixed pins: i = Z_c."""
        return self.Z_c

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise KeyError(f"Unknown parameters in JSON: {unknown}")
        return cls(**{k: v for k, v in data.items()})