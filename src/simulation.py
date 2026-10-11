"""Shared engine for CLI, UI and ensembles. Particle snapshots are optional."""
from dataclasses import asdict
from pathlib import Path
import json
import platform
import subprocess
import time
from importlib.metadata import version
import h5py
import numpy as np
from galaxy import build_initial_conditions
from integrator import leapfrog_step, leapfrog_step_direct, compute_energy
from octree import compute_accelerations, compute_accelerations_direct
from .metrics import center, encounter_metrics, diagnostics

def simulate(config, retain_particles=True):
    started=time.perf_counter()
    pos,vel,mass=build_initial_conditions(config)
    n=config.N
    direct=config.method=="direct" or (config.method=="auto" and 2*n<=3000)
    force=compute_accelerations_direct if direct else compute_accelerations
    stepper=leapfrog_step_direct if direct else leapfrog_step
    kwargs={"softening":config.softening}
    if not direct: kwargs["theta"]=config.theta
    acc=force(pos,mass,**kwargs)
    c0,v0=center(pos,mass),center(vel,mass)
    # Initial total momentum is nearly zero in the barycentric frame.
    # Normalize by sum(m_i |v_i|), never by that near-zero vector.
    pscale=float(np.sum(mass*np.linalg.norm(vel,axis=1)))
    records={k:[] for k in ("time","com_mw","com_m31","separation","momentum",
                            "momentum_drift","com_drift_kpc")}
    if config.validate: records["energy"]=[]
    snapshots=[]; velocities=[]
    for step in range(config.steps+1):
        if step:
            pos,vel,acc=stepper(pos,vel,acc,mass,dt=config.dt,**kwargs)
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise FloatingPointError(f"Non-finite state at step {step}")
        if step % config.sample_every and step != config.steps: continue
        t=step*config.dt
        mw,m31=center(pos[:n],mass[:n]),center(pos[n:],mass[n:])
        momentum,pdrift,cdrift=diagnostics(pos,vel,mass,c0,v0,t,pscale)
        for key,value in dict(time=t,com_mw=mw,com_m31=m31,separation=np.linalg.norm(m31-mw),
                              momentum=momentum,momentum_drift=pdrift,com_drift_kpc=cdrift).items():
            records[key].append(value)
        if config.validate: records["energy"].append(sum(compute_energy(pos,vel,mass,config.softening)))
        if retain_particles:
            snapshots.append(pos.copy()); velocities.append(vel.copy())
    data={k:np.asarray(v) for k,v in records.items()}
    data["mass"]=mass
    if retain_particles:
        data.update(pos=np.asarray(snapshots),vel=np.asarray(velocities))
    metrics=encounter_metrics(data["time"],data["separation"],config.close_threshold_kpc)
    if config.validate:
        e0=abs(data["energy"][0])
        if e0 < np.finfo(float).tiny: raise ValueError("Initial total energy too small for relative drift")
        data["energy_drift"]=(data["energy"]-data["energy"][0])/e0
    metrics.update(max_energy_drift_pct=float(100*np.max(np.abs(data["energy_drift"]))) if config.validate else None,
                   max_momentum_drift_relative=float(data["momentum_drift"].max()),
                   max_com_drift_kpc=float(data["com_drift_kpc"].max()),
                   momentum_scale=pscale, duration_gyr=config.dt*config.steps/1e9,
                   runtime_s=time.perf_counter()-started,method_used="direct" if direct else "bh",
                   sampled_diagnostics=True)
    return data,metrics

def save_run(path,config,data,metrics):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    provenance=run_provenance()
    with h5py.File(path,"w") as hf:
        hf.attrs["config_json"]=json.dumps(asdict(config))
        hf.attrs["metrics_json"]=json.dumps(metrics)
        hf.attrs["provenance_json"]=json.dumps(provenance)
        for key,value in data.items(): hf.create_dataset(key,data=value,compression="gzip")
        hf["time"].attrs["unit"]="yr"
        for key in ("pos","com_mw","com_m31","separation","com_drift_kpc"):
            if key in hf: hf[key].attrs["unit"]="kpc"
        if "vel" in hf: hf["vel"].attrs["unit"]="kpc/yr"
        hf["mass"].attrs["unit"]="solar_mass"
        hf["momentum"].attrs["unit"]="solar_mass*kpc/yr"
        if "energy" in hf: hf["energy"].attrs["unit"]="solar_mass*kpc^2/yr^2"
    for name,value in (("config",asdict(config)),("metrics",metrics),("provenance",provenance)):
        path.with_name(path.stem+"."+name+".json").write_text(json.dumps(value,indent=2),encoding="utf-8")

def load_summary(path):
    with h5py.File(path,"r") as hf:
        keys=("time","com_mw","com_m31","separation","energy_drift","momentum_drift","com_drift_kpc")
        data={k:hf[k][:] for k in keys if k in hf}
        return data,json.loads(hf.attrs["metrics_json"]),json.loads(hf.attrs["config_json"])


def run_provenance():
    root=Path(__file__).resolve().parents[1]
    def git(*args):
        try: return subprocess.check_output(["git",*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): return "unavailable"
    import hashlib
    sources=sorted(root.glob("*.py"))+sorted((root/"src").glob("*.py"))
    provenance=dict(schema_version=2,git_commit=git("rev-parse","HEAD"),
                    git_dirty=bool(git("status","--porcelain")),python=platform.python_version(),
                    versions={n:version(n) for n in ("numpy","h5py","matplotlib","streamlit")},
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    return provenance
