"""Single source of simulation defaults; presets are illustrative experiments."""
from dataclasses import dataclass, asdict, replace
import argparse
import json
import math

G_CONST = 4.498e-24
KMS_TO_KPCYR = 1.022712e-9

@dataclass(frozen=True)
class Config:
    N: int = 80  # Per galaxy, preserving the original CLI convention.
    dt: float = 2e6
    steps: int = 3000
    softening: float = 5.0
    method: str = "auto"
    theta: float = 0.5
    sample_every: int = 20
    frame_every: int = 3
    seed: int = 42
    mw_mass: float = 1.06e12
    m31_mass: float = 1.60e12
    initial_distance_kpc: float = 785.0
    andromeda_radial_kms: float = -110.0
    andromeda_transverse_kms: float = 17.0
    mw_radius_kpc: float = 20.0
    m31_radius_kpc: float = 25.0
    mw_inclination_deg: float = 77.0
    m31_inclination_deg: float = 0.0
    dispersion: float = 0.15
    close_threshold_kpc: float = 100.0
    validate: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("N","steps","sample_every","frame_every","seed"):
            if not isinstance(getattr(self,key),int):
                raise ValueError(f"{key} must be an integer")
        if self.N < 10 or self.seed < 0:
            raise ValueError("N must be >=10 per galaxy and seed >=0")
        for key in ("dt","steps","softening","theta","sample_every","frame_every",
                    "mw_mass","m31_mass","initial_distance_kpc","mw_radius_kpc",
                    "m31_radius_kpc","close_threshold_kpc"):
            if getattr(self,key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.dispersion < 0 or self.method not in ("auto","direct","bh"):
            raise ValueError("Invalid dispersion or method")

PRESETS = {
    "current_estimate": {},
    "head_on": {"andromeda_transverse_kms": 0.0},
    "high_transverse": {"andromeda_transverse_kms": 80.0},
    "low_transverse": {"andromeda_transverse_kms": 8.0},
    "high_mass_andromeda": {"m31_mass": 2.0e12},
    "high_mass_milky_way": {"mw_mass": 1.8e12},
}

def preset_config(name="current_estimate", **overrides):
    return Config(**(PRESETS[name] | overrides))

def parser(description):
    p=argparse.ArgumentParser(description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset",choices=PRESETS,default="current_estimate",
                   help="Illustrative scenario; not a current observational inference")
    p.add_argument("--config",help="Load a saved config JSON; explicit flags take priority")
    units={"N":"particles per galaxy (total = 2N)","dt":"yr","steps":"integration steps",
           "softening":"kpc","sample_every":"steps between saved diagnostics / snapshots",
           "frame_every":"saved snapshots between animation frames","seed":"random seed",
           "mw_mass":"solar masses","m31_mass":"solar masses","initial_distance_kpc":"kpc",
           "andromeda_radial_kms":"km/s; negative = approaching",
           "andromeda_transverse_kms":"km/s","mw_radius_kpc":"halo scale radius, kpc",
           "m31_radius_kpc":"halo scale radius, kpc","mw_inclination_deg":"degrees",
           "m31_inclination_deg":"degrees","dispersion":"fraction of circular speed",
           "close_threshold_kpc":"kpc","theta":"Barnes-Hut opening angle"}
    for key,default in asdict(Config()).items():
        if key in ("method","validate"): continue
        p.add_argument("--"+key.replace("_","-"),dest=key,type=type(default),default=None,
                       help=f"{units[key]}; baseline {default}")
    p.add_argument("--method",choices=["auto","direct","bh"],default=None)
    p.add_argument("--validate",action=argparse.BooleanOptionalAction,default=None,
                   help="Energy diagnostics (enabled by default); momentum/COM always saved")
    p.add_argument("--duration-gyr",type=float,help="Replace steps; must be an integer multiple of dt")
    return p

def resolve(args):
    values=PRESETS[args.preset].copy()
    if args.config:
        with open(args.config,encoding="utf-8") as stream:
            values.update(json.load(stream))
    values.update({k:v for k,v in vars(args).items() if k in asdict(Config()) and v is not None})
    if args.duration_gyr is not None:
        count=args.duration_gyr*1e9/values.get("dt",Config().dt)
        if not math.isfinite(count) or not math.isclose(count,round(count),abs_tol=1e-7):
            raise ValueError("duration must be an integer multiple of dt")
        values["steps"]=round(count)
    return Config(**values)
